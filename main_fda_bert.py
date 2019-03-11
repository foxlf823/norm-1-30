
import random

import data
import train
from my_utils import makedir_and_clear

import logging
import torch.nn as nn
from alphabet import Alphabet
from options import opt


import torch
from torch.utils.data import DataLoader, Dataset
import time
import os
from data_structure import Entity
import numpy as np


from pytorch_pretrained_bert import BertTokenizer, BertAdam, BertModel
from pytorch_pretrained_bert.modeling import BertPreTrainedModel
from data import get_fda_file
from data_structure import Document
tokenizer = BertTokenizer.from_pretrained(opt.bert_dir)
import codecs
import re

from stopword import stop_word


def load_meddra_dict(data):
    input_path = data.config['norm_dict']

    map_id_to_name = dict()

    with codecs.open(input_path, 'r', 'UTF-8') as fp:
        for line in fp:
            line = line.strip()
            if line == u'':
                continue
            token = re.split(r"\|\|", line)
            cui = token[0]

            conceptNames = token[1]

            map_id_to_name[cui] = conceptNames


    return map_id_to_name


def pad_sequence(x, max_len):

    padded_x = np.zeros((len(x), max_len), dtype=np.int)
    for i, row in enumerate(x):
        padded_x[i][:len(row)] = row

    padded_x = torch.LongTensor(padded_x)

    return padded_x

def my_collate(batch):
    x, y = zip(*batch)

    tokens = [x_['entity'] for x_ in x]
    lengths = [len(x_['entity']) for x_ in x]
    max_len = max(lengths)
    mask = [[1] * length for length in lengths]
    sentences = [x_['sentence'] for x_ in x]

    tokens = pad_sequence(tokens, max_len)
    mask = pad_sequence(mask, max_len)
    sentences = pad_sequence(sentences, max_len)

    y = torch.LongTensor(y).view(-1)

    if opt.gpu >= 0 and torch.cuda.is_available():
        tokens = tokens.cuda(opt.gpu)
        mask = mask.cuda(opt.gpu)
        sentences = sentences.cuda(opt.gpu)
        y = y.cuda(opt.gpu)
    return tokens, mask, sentences, y


class MyDataset(Dataset):

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

        assert len(self.X) == len(self.Y), 'X and Y have different lengths'

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx])


def get_dict_index(dict_alphabet, concept_id):
    index = dict_alphabet.get_index(concept_id)-2 # since alphabet begin at 2
    return index

def get_dict_name(dict_alphabet, concept_index):
    name = dict_alphabet.get_instance(concept_index+2)
    return name

def init_dict_alphabet(dict_alphabet, dictionary):
    # concept_name may be string or umls_concept
    for concept_id, concept_name in dictionary.items():
        dict_alphabet.add(concept_id)

def get_dict_size(dict_alphabet):
    return dict_alphabet.size()-2

def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word

def word_preprocess(word):
    if opt.norm_number_normalized:
        word = normalize_word(word)
    word = word.lower()
    return word

def generate_instances(entities, dict_alphabet):
    Xs = []
    Ys = []

    for entity in entities:
        if len(entity.norm_ids) > 0:
            Y = get_dict_index(dict_alphabet, entity.norm_ids[0])  # use the first id to generate instance
            if Y >= 0 and Y < get_dict_size(dict_alphabet):  # for tac, can be none or oov ID
                pass
            else:
                continue
        else:
            Y = 0

        X = {}

        tokens = []
        for token in tokenizer.tokenize(entity.name):
            if token in stop_word:
                continue
            token = word_preprocess(token)
            tokens.append(token)

        tokens.insert(0, '[CLS]')
        tokens.append('[SEP]')
        word_ids = tokenizer.convert_tokens_to_ids(tokens)

        if len(word_ids) == 0:
            continue


        X['entity'] = word_ids
        X['sentence'] = [0]*len(word_ids)

        Xs.append(X)
        Ys.append(Y)

    return Xs, Ys


def processOneFile_fda(fileName, annotation_dir, types, type_filter, isFDA2018, isNorm):
    documents = []
    annotation_file = get_fda_file(os.path.join(annotation_dir, fileName))

    # each section is a document
    for section in annotation_file.sections:
        document = Document()
        document.name = fileName[:fileName.find('.')]+"_"+section.id
        if section.text is None:
            document.text = ""
            document.entities = []
            document.sentences = []
            documents.append(document)
            continue

        document.text = section.text

        entities = []

        if isFDA2018==False and isNorm==True:
            for reaction in annotation_file.reactions:
                entity = Entity()
                entity.name = reaction.name
                for normalization in reaction.normalizations:
                    entity.norm_ids.append(normalization.meddra_pt_id) # can be none
                    entity.norm_names.append(normalization.meddra_pt)
                entities.append(entity)

        else:
            for entity in annotation_file.mentions:
                if entity.section != section.id:
                    continue
                if types and (entity.type not in type_filter):
                    continue
                entities.append(entity)

        document.entities = entities

        document.sentences = []

        documents.append(document)

    return documents, annotation_file


def load_data_fda(basedir, types, type_filter, isFDA2018, isNorm):

    logging.info("load_data_fda: {}".format(basedir))


    documents = []

    count_document = 0
    count_section = 0
    count_sentence = 0
    count_entity = 0

    annotation_files = [f for f in os.listdir(basedir) if f.find('.xml')!=-1]
    for fileName in annotation_files:
        try:
            document, _ = processOneFile_fda(fileName, basedir, types, type_filter, isFDA2018, isNorm)
        except Exception as e:
            logging.error("process file {} error: {}".format(fileName, e))
            continue

        documents.extend(document)

        # statistics
        count_document += 1
        for d in document:
            count_section += 1
            count_sentence += len(d.sentences)
            count_entity += len(d.entities)

    logging.info("document number: {}".format(count_document))
    logging.info("section number: {}".format(count_section))
    logging.info("sentence number: {}".format(count_sentence))
    logging.info("entity number {}".format(count_entity))

    return documents



class BertForSequenceClassification(BertPreTrainedModel):


    def __init__(self, config, target):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = get_dict_size(target)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.apply(self.init_bert_weights)
        self.dict_alphabet = target
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits

    def loss(self, y_pred, y_gold):

        return self.criterion(y_pred, y_gold)

    def process_one_doc(self, doc, entities, dictionary, dictionary_reverse):

        Xs, Ys = generate_instances(entities, self.dict_alphabet)

        data_loader = DataLoader(MyDataset(Xs, Ys), opt.batch_size, shuffle=False, collate_fn=my_collate)
        data_iter = iter(data_loader)
        num_iter = len(data_loader)

        entity_start = 0

        for i in range(num_iter):

            x, mask, sentences, _ = next(data_iter)

            y_pred = self.forward(x, sentences, mask)

            values, indices = torch.max(y_pred, 1)

            actual_batch_size = x.size(0)

            for batch_idx in range(actual_batch_size):
                entity = entities[entity_start+batch_idx]
                norm_id = get_dict_name(self.dict_alphabet, indices[batch_idx].item())

                name = dictionary[norm_id]
                entity.norm_ids.append(norm_id)
                entity.norm_names.append(name)

            entity_start += actual_batch_size

def evaluate_for_fda(gold_entities, pred_entities):

    ct_gold = len(gold_entities)
    ct_predicted = len(pred_entities)
    ct_correct = 0
    for idx, pred in enumerate(pred_entities):
        gold = gold_entities[idx]
        if len(pred.norm_ids) != 0 and pred.norm_ids[0] in gold.norm_ids:
            ct_correct += 1

    return ct_gold, ct_predicted, ct_correct

def evaluate(documents, dictionary, dictionary_reverse, model):
    model.eval()

    ct_predicted = 0
    ct_gold = 0
    ct_correct = 0

    for document in documents:

        # copy entities from gold entities
        pred_entities = []
        for gold in document.entities:
            pred = Entity()
            pred.id = gold.id
            pred.type = gold.type
            pred.spans = gold.spans
            pred.section = gold.section
            pred.name = gold.name
            pred_entities.append(pred)


        model.process_one_doc(document, pred_entities, dictionary, dictionary_reverse)


        p1, p2, p3 = evaluate_for_fda(document.entities, pred_entities)

        ct_gold += p1
        ct_predicted += p2
        ct_correct += p3


    if ct_gold == 0:
        precision = 0
        recall = 0
    else:
        precision = ct_correct * 1.0 / ct_predicted
        recall = ct_correct * 1.0 / ct_gold

    if precision+recall == 0:
        f_measure = 0
    else:
        f_measure = 2*precision*recall/(precision+recall)

    return precision, recall, f_measure


def train(train_data, dev_data, test_data, d, dictionary, dictionary_reverse, opt, fold_idx, isMeddra_dict):
    logging.info("train the neural-based normalization model ...")

    external_train_data = []
    if d.config.get('norm_ext_corpus') is not None:
        for k, v in d.config['norm_ext_corpus'].items():
            if k == 'tac':
                external_train_data.extend(load_data_fda(v['path'], True, v.get('types'), v.get('types'), False, True))
            else:
                raise RuntimeError("not support external corpus")
    if len(external_train_data) != 0:
        train_data.extend(external_train_data)

    logging.info("build alphabet ...")
    # word_alphabet = Alphabet('word')
    # build_alphabet_from_dict(word_alphabet, dictionary)
    # build_alphabet(word_alphabet, train_data)
    # if opt.dev_file:
    #     build_alphabet(word_alphabet, dev_data)
    # if opt.test_file:
    #     build_alphabet(word_alphabet, test_data)
    # norm_utils.fix_alphabet(word_alphabet)
    # logging.info("alphabet size {}".format(word_alphabet.size()))


    dict_alphabet = Alphabet('dict')
    init_dict_alphabet(dict_alphabet, dictionary)
    dict_alphabet.close()

    train_X = []
    train_Y = []
    for doc in train_data:


        temp_X, temp_Y = generate_instances(doc.entities, dict_alphabet)

        train_X.extend(temp_X)
        train_Y.extend(temp_Y)


    train_loader = DataLoader(MyDataset(train_X, train_Y), opt.batch_size, shuffle=True, collate_fn=my_collate)


    if opt.gpu >= 0 and torch.cuda.is_available():
        device = torch.device('cuda', opt.gpu)
    else:
        device = torch.device('cpu')

    model = BertForSequenceClassification.from_pretrained(opt.bert_dir,
                                                          target=dict_alphabet)
    model.dict_alphabet = dict_alphabet
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = BertAdam(optimizer_grouped_parameters, lr=opt.lr)

    best_dev_f = -10
    best_dev_p = -10
    best_dev_r = -10

    bad_counter = 0

    logging.info("start training ...")

    for idx in range(opt.iter):
        epoch_start = time.time()

        model.train()

        train_iter = iter(train_loader)
        num_iter = len(train_loader)

        sum_loss = 0

        correct, total = 0, 0

        for i in range(num_iter):

            x, mask, sentences, y = next(train_iter)

            y_pred = model.forward(x, sentences, mask)

            l = model.loss(y_pred, y)

            sum_loss += l.item()

            l.backward()
            optimizer.step()
            model.zero_grad()

            total += y.size(0)
            _, pred = torch.max(y_pred, 1)
            correct += (pred == y).sum().item()

        epoch_finish = time.time()
        accuracy = 100.0 * correct / total
        logging.info("epoch: %s training finished. Time: %.2fs. loss: %.4f Accuracy %.2f" % (
        idx, epoch_finish - epoch_start, sum_loss / num_iter, accuracy))

        if opt.dev_file:
            p, r, f = evaluate(dev_data, dictionary, dictionary_reverse, model)
            logging.info("Dev: p: %.4f, r: %.4f, f: %.4f" % (p, r, f))
        else:
            f = best_dev_f

        if f > best_dev_f:
            logging.info("Exceed previous best f score on dev: %.4f" % (best_dev_f))

            best_dev_f = f
            best_dev_p = p
            best_dev_r = r

            bad_counter = 0
        else:
            bad_counter += 1

        if len(opt.dev_file) != 0 and bad_counter >= opt.patience:
            logging.info('Early Stop!')
            break

    logging.info("train finished")

    return best_dev_p, best_dev_r, best_dev_f

def main():

    logger = logging.getLogger()
    if opt.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    logging.info(opt)

    if opt.random_seed != 0:
        random.seed(opt.random_seed)
        np.random.seed(opt.random_seed)
        torch.manual_seed(opt.random_seed)
        torch.cuda.manual_seed_all(opt.random_seed)


    d = data.Data(opt)
    logging.info(d.config)

    makedir_and_clear(opt.output)


    documents = load_data_fda(opt.train_file, opt.types, opt.type_filter, True, True)


    logging.info("use {} fold cross validataion".format(opt.cross_validation))
    fold_num = opt.cross_validation
    total_doc_num = len(documents)
    dev_doc_num = total_doc_num // fold_num

    macro_p = 0.0
    macro_r = 0.0
    macro_f = 0.0

    meddra_dict = load_meddra_dict(d)


    for fold_idx in range(fold_num):

        fold_start = fold_idx*dev_doc_num
        fold_end = fold_idx*dev_doc_num+dev_doc_num
        if fold_end > total_doc_num:
            fold_end = total_doc_num
        if fold_idx == fold_num-1 and fold_end < total_doc_num:
            fold_end = total_doc_num

        train_data = []
        train_data.extend(documents[:fold_start])
        train_data.extend(documents[fold_end:])
        dev_data = documents[fold_start:fold_end]

        logging.info("begin fold {}".format(fold_idx))
        logging.info("doc start {}, doc end {}".format(fold_start, fold_end))


        p, r, f = train(train_data, dev_data, None, d, meddra_dict, None, opt, fold_idx, True)

        macro_p += p
        macro_r += r
        macro_f += f


    logging.info("the macro averaged p r f are %.4f, %.4f, %.4f" % (macro_p*1.0/fold_num, macro_r*1.0/fold_num, macro_f*1.0/fold_num))



if __name__ == '__main__':
    main()

