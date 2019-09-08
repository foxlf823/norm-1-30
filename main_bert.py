
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
import umls


from pytorch_pretrained_bert import BertTokenizer, BertAdam, BertModel
from pytorch_pretrained_bert.modeling import BertPreTrainedModel
from data import get_fda_file
from data_structure import Document
tokenizer = BertTokenizer.from_pretrained(opt.bert_dir)
import codecs
import re
import torch.nn.functional as functional

from stopword import stop_word


def init_dict_alphabet(dict_alphabet, dictionary):
    # concept_name may be string or umls_concept
    for concept_id, concept_name in dictionary.items():
        dict_alphabet.add(concept_id)

def get_dict_index(dict_alphabet, concept_id):
    index = dict_alphabet.get_index(concept_id)-2 # since alphabet begin at 2
    return index

def get_dict_size(dict_alphabet):
    return dict_alphabet.size()-2

def get_dict_name(dict_alphabet, concept_index):
    name = dict_alphabet.get_instance(concept_index+2)
    return name

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

def generate_instances_ehr(entities, dict_alphabet, dictionary_reverse):
    Xs = []
    Ys = []

    for entity in entities:
        if len(entity.norm_ids) > 0:
            if entity.norm_ids[0] in dictionary_reverse:
                cui_list = dictionary_reverse[entity.norm_ids[0]]
                Y = get_dict_index(dict_alphabet, cui_list[0])  # use the first id to generate instance
                if Y >= 0 and Y < get_dict_size(dict_alphabet):
                    Ys.append(Y)
                else:
                    raise RuntimeError("entity {}, {}, cui not in dict_alphabet".format(entity.id, entity.name))
            else:
                logging.debug("entity {}, {}, can't map to umls, ignored".format(entity.id, entity.name))
                continue
        else:
            Ys.append(0)


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

    return Xs, Ys

class MyDataset(Dataset):

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

        assert len(self.X) == len(self.Y), 'X and Y have different lengths'

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx])


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

    lengths = torch.LongTensor(lengths)

    if opt.gpu >= 0 and torch.cuda.is_available():
        tokens = tokens.cuda(opt.gpu)
        mask = mask.cuda(opt.gpu)
        sentences = sentences.cuda(opt.gpu)
        y = y.cuda(opt.gpu)
    return tokens, mask, sentences, y, lengths


class DotAttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(DotAttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(hidden_size, 1, bias=False)
        self.W.weight.data.copy_(torch.ones((1, hidden_size)))

    def forward(self, inputs, lengths):
        """
        input: (unpacked_padded_output: batch_size x seq_len x hidden_size, lengths: batch_size)
        """
        batch_size, max_len, _ = inputs.size()
        flat_input = inputs.contiguous().view(-1, self.hidden_size)
        logits = self.W(flat_input).view(batch_size, max_len)
        alphas = functional.softmax(logits, dim=1)

        # computing mask
        idxes = torch.arange(0, max_len, out=torch.LongTensor(max_len)).unsqueeze(0)
        if opt.gpu >= 0 and torch.cuda.is_available():
            idxes = idxes.cuda(opt.gpu)
        mask = (idxes<lengths.unsqueeze(1)).float()

        alphas = alphas * mask
        # renormalize
        alphas = alphas / torch.sum(alphas, 1).view(-1, 1)

        return alphas

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
        encoded_layers, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return encoded_layers, logits

    def loss(self, y_pred, y_gold):

        return self.criterion(y_pred, y_gold)

    def process_one_doc(self, doc, entities, dictionary, dictionary_reverse):

        Xs, Ys = generate_instances_ehr(entities, self.dict_alphabet, dictionary_reverse)

        data_loader = DataLoader(MyDataset(Xs, Ys), opt.batch_size, shuffle=False, collate_fn=my_collate)
        data_iter = iter(data_loader)
        num_iter = len(data_loader)

        entity_start = 0

        for i in range(num_iter):

            x, mask, sentences, _, _ = next(data_iter)

            _, y_pred = self.forward(x, sentences, mask)

            values, indices = torch.max(y_pred, 1)

            actual_batch_size = x.size(0)

            for batch_idx in range(actual_batch_size):
                entity = entities[entity_start+batch_idx]
                norm_id = get_dict_name(self.dict_alphabet, indices[batch_idx].item())

                concept = dictionary[norm_id]
                entity.norm_ids.append(norm_id)
                entity.norm_names.append(concept.names)

            entity_start += actual_batch_size

from collections import Counter
def evaluate_for_ehr(gold_entities, pred_entities, dictionary, metamap_entities, correct_counter, wrong_counter):

    ct_norm_gold = len(gold_entities)
    ct_norm_predict = len(pred_entities)
    ct_norm_correct = 0



    for predict_entity in pred_entities:

        for gold_entity in gold_entities:

            if predict_entity.equals_span(gold_entity):

                if len(gold_entity.norm_ids) == 0:
                    # if gold_entity not annotated, we count it as TP
                    ct_norm_correct += 1
                else:

                    if len(predict_entity.norm_ids) != 0 and predict_entity.norm_ids[0] in dictionary:
                        concept = dictionary[predict_entity.norm_ids[0]]

                        if gold_entity.norm_ids[0] in concept.codes:
                            ct_norm_correct += 1
                            for metamap_entity in metamap_entities:
                                if metamap_entity.equals_span(predict_entity):
                                    if len(metamap_entity.norm_ids) != 0 and metamap_entity.norm_ids[0] in dictionary:
                                        metamap_concept = dictionary[metamap_entity.norm_ids[0]]
                                        if gold_entity.norm_ids[0] not in metamap_concept.codes:
                                            # print("bert right : {} {} {} | metamap wrong: {} {}".format(
                                            #     gold_entity.name, gold_entity.norm_ids[0],
                                            #     gold_entity.norm_names[0], metamap_concept.codes,
                                            #     metamap_concept.names))
                                            pass

                            term_name = predict_entity.name.lower()
                            if term_name in correct_counter:
                                correct_counter[term_name] += 1
                            else:
                                correct_counter[term_name] = 1

                        else:
                            # print("gold: {} {} {} | pred: {} {}".format(
                            #     gold_entity.name, gold_entity.norm_ids[0],
                            #     gold_entity.norm_names[0], concept.codes,
                            #     concept.names))
                            for metamap_entity in metamap_entities:
                                if metamap_entity.equals_span(predict_entity):
                                    if len(metamap_entity.norm_ids) != 0 and metamap_entity.norm_ids[0] in dictionary:
                                        metamap_concept = dictionary[metamap_entity.norm_ids[0]]
                                        if gold_entity.norm_ids[0] in metamap_concept.codes:
                                            # print("metamap right : {} {} {} | bert wrong: {} {}".format(
                                            #     gold_entity.name, gold_entity.norm_ids[0],
                                            #     gold_entity.norm_names[0], concept.codes,
                                            #     concept.names))
                                            pass

                            term_name = predict_entity.name.lower()
                            if term_name in wrong_counter:
                                wrong_counter[term_name] += 1
                            else:
                                wrong_counter[term_name] = 1


                break

    return ct_norm_gold, ct_norm_predict, ct_norm_correct

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

        p1, p2, p3 = evaluate_for_ehr(document.entities, pred_entities, dictionary)

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

    logging.info("build alphabet ...")


    dict_alphabet = Alphabet('dict')
    init_dict_alphabet(dict_alphabet, dictionary)
    dict_alphabet.close()

    train_X = []
    train_Y = []
    for doc in train_data:

        temp_X, temp_Y = generate_instances_ehr(doc.entities, dict_alphabet, dictionary_reverse)
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

            x, mask, sentences, y, _ = next(train_iter)

            _, y_pred = model.forward(x, sentences, mask)

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

            torch.save(model, os.path.join(opt.output, "norm_neural.pkl"))

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

from my_utils import get_bioc_file, get_text_file
import metamap
type_we_care = set(['ADE', 'SSLIF', 'Indication'])

def parse_one_gold_file(annotation_dir, corpus_dir, fileName):
    document = Document()
    document.name = fileName[:fileName.find('.')]

    annotation_file = get_bioc_file(os.path.join(annotation_dir, fileName))
    bioc_passage = annotation_file[0].passages[0]
    entities = []

    for entity in bioc_passage.annotations:
        if entity.infons['type'] not in type_we_care:
            continue

        entity_ = Entity()
        entity_.id = entity.id
        processed_name = entity.text.replace('\\n', ' ')
        if len(processed_name) == 0:
            logging.debug("{}: entity {} name is empty".format(fileName, entity.id))
            continue
        entity_.name = processed_name
        entity_.type = entity.infons['type']
        entity_.spans.append([entity.locations[0].offset, entity.locations[0].end])

        if ('SNOMED code' in entity.infons and entity.infons['SNOMED code'] != 'N/A') \
                and ('SNOMED term' in entity.infons and entity.infons['SNOMED term'] != 'N/A'):
            entity_.norm_ids.append(entity.infons['SNOMED code'])
            entity_.norm_names.append(entity.infons['SNOMED term'])

        elif ('MedDRA code' in entity.infons and entity.infons['MedDRA code'] != 'N/A') \
                and ('MedDRA term' in entity.infons and entity.infons['MedDRA term'] != 'N/A'):
            entity_.norm_ids.append(entity.infons['MedDRA code'])
            entity_.norm_names.append(entity.infons['MedDRA term'])
        else:
            logging.debug("{}: no norm id in entity {}".format(fileName, entity.id))
            # some entities may have no norm id
            continue

        entities.append(entity_)

    document.entities = entities

    corpus_file = get_text_file(os.path.join(corpus_dir, fileName.split('.bioc')[0]))
    document.text = corpus_file

    return document


if opt.whattodo == 1:

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

    logging.info("load data ...")
    train_data = data.loadData(opt.train_file, True, opt.types, opt.type_filter)
    dev_data = data.loadData(opt.dev_file, True, opt.types, opt.type_filter)
    if opt.test_file:
        test_data = data.loadData(opt.test_file, False, opt.types, opt.type_filter)
    else:
        test_data = None

    logging.info("load dict ...")
    UMLS_dict, UMLS_dict_reverse = umls.load_umls_MRCONSO(d.config['norm_dict'])
    logging.info("dict concept number {}".format(len(UMLS_dict)))

    train(train_data, dev_data, test_data, d, UMLS_dict, UMLS_dict_reverse, opt, None, False)

elif opt.whattodo == 2:



    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    from collections import OrderedDict
    import json
    def metamap_ner_my_norm(d):
        print("load umls ...")

        UMLS_dict, UMLS_dict_reverse = umls.load_umls_MRCONSO(d.config['norm_dict'])

        predict_dir = "/Users/feili/Desktop/umass/CancerADE_SnoM_30Oct2017_test/metamap"
        annotation_dir = os.path.join(opt.test_file, 'bioc')
        corpus_dir = os.path.join(opt.test_file, 'txt')
        annotation_files = [f for f in os.listdir(annotation_dir) if os.path.isfile(os.path.join(annotation_dir, f))]



        if opt.test_in_cpu:
            model = torch.load(os.path.join(opt.output, 'norm_neural.pkl'), map_location='cpu')
        else:
            model = torch.load(os.path.join(opt.output, 'norm_neural.pkl'))
        model.eval()


        ct_norm_predict = 0
        ct_norm_gold = 0
        ct_norm_correct = 0

        correct_counter = Counter()
        wrong_counter = Counter()

        for gold_file_name in annotation_files:
            print("# begin {}".format(gold_file_name))
            gold_document = parse_one_gold_file(annotation_dir, corpus_dir, gold_file_name)

            predict_document = metamap.load_metamap_result_from_file(
                os.path.join(predict_dir, gold_file_name[:gold_file_name.find('.')] + ".field.txt"))

            # copy entities from metamap entities
            pred_entities = []
            for gold in predict_document.entities:
                pred = Entity()
                pred.id = gold.id
                pred.type = gold.type
                pred.spans = gold.spans
                pred.section = gold.section
                pred.name = gold.name
                pred_entities.append(pred)


            model.process_one_doc(gold_document, pred_entities, UMLS_dict, UMLS_dict_reverse)


            p1, p2, p3 = evaluate_for_ehr(gold_document.entities, pred_entities, UMLS_dict,
                                          predict_document.entities, correct_counter, wrong_counter)

            ct_norm_gold += p1
            ct_norm_predict += p2
            ct_norm_correct += p3




        sorted_correct_entities = OrderedDict(correct_counter.most_common())
        sorted_correct_entities = json.dumps(sorted_correct_entities, indent=4)
        with codecs.open("sorted_correct_entities.txt", 'w', 'UTF-8') as fp:
            fp.write(sorted_correct_entities)

        sorted_wrong_entities = OrderedDict(wrong_counter.most_common())
        sorted_wrong_entities = json.dumps(sorted_wrong_entities, indent=4)
        with codecs.open("sorted_wrong_entities.txt", 'w', 'UTF-8') as fp:
            fp.write(sorted_wrong_entities)

        p = ct_norm_correct * 1.0 / ct_norm_predict
        r = ct_norm_correct * 1.0 / ct_norm_gold
        f1 = 2.0 * p * r / (p + r)
        print("NORM p: %.4f | r: %.4f | f1: %.4f" % (p, r, f1))


    d = data.Data(opt)

    metamap_ner_my_norm(d)

else:
    import matplotlib.pyplot as plt
    def make_heatmaps(model, tokens, attention_values, cmap_val, words_or_sent=True, fig_size_v=10, model_name="Model"):
        models = [model]
        attention_values = np.array([attention_values])
        fig, ax = plt.subplots()
        im = ax.imshow(attention_values, cmap=cmap_val)

        # We want to show all ticks...
        ax.set_xticks(np.arange(len(tokens)))
        # ax.set_yticks(np.arange(len(models)))
        plt.yticks([])
        # ... and label them with the respective list entries
        ax.set_xticklabels(tokens, fontsize=14)
        # ax.set_yticklabels(models)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # ax.set_title(model_name)
        if (words_or_sent):
            fig.set_size_inches(fig_size_v, 2)
        else:
            fig.set_size_inches(7, 2)
        plt.show()

    logger = logging.getLogger()
    if opt.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    logging.info(opt)

    d = data.Data(opt)

    logging.info(d.config)

    logging.info("load dict ...")
    UMLS_dict, UMLS_dict_reverse = umls.load_umls_MRCONSO(d.config['norm_dict'])

    annotation_dir = os.path.join(opt.test_file, 'bioc')
    corpus_dir = os.path.join(opt.test_file, 'txt')
    annotation_files = [f for f in os.listdir(annotation_dir) if os.path.isfile(os.path.join(annotation_dir, f))]

    tokenizer = BertTokenizer.from_pretrained(opt.bert_dir)

    if opt.test_in_cpu:
        model = torch.load(os.path.join(opt.output, 'norm_neural.pkl'), map_location='cpu')
    else:
        model = torch.load(os.path.join(opt.output, 'norm_neural.pkl'))
    model.eval()

    attention_model = DotAttentionLayer(768)
    file_count = 0
    for gold_file_name in annotation_files:
        print("# begin {}".format(gold_file_name))

        if file_count < 1:
            file_count += 1
            continue

        file_count += 1

        gold_document = parse_one_gold_file(annotation_dir, corpus_dir, gold_file_name)

        pred_entities = []
        for gold in gold_document.entities:
            pred = Entity()
            pred.id = gold.id
            pred.type = gold.type
            pred.spans = gold.spans
            pred.section = gold.section
            pred.name = gold.name
            pred_entities.append(pred)

        Xs, Ys = generate_instances_ehr(pred_entities, model.dict_alphabet, UMLS_dict_reverse)

        data_loader = DataLoader(MyDataset(Xs, Ys), opt.batch_size, shuffle=False, collate_fn=my_collate)
        data_iter = iter(data_loader)
        num_iter = len(data_loader)

        entity_start = 0

        for i in range(num_iter):

            x, mask, sentences, _, lengths = next(data_iter)

            encoded_layers, y_pred = model.forward(x, sentences, mask)

            values, indices = torch.max(y_pred, 1)

            actual_batch_size = x.size(0)

            if entity_start < 35:
                entity_start += actual_batch_size
                continue
            elif entity_start > 40:
                break

            for batch_idx in range(actual_batch_size):
                predict_entity = pred_entities[entity_start + batch_idx]
                norm_id = get_dict_name(model.dict_alphabet, indices[batch_idx].item())

                concept = UMLS_dict[norm_id]
                predict_entity.norm_ids.append(norm_id)
                predict_entity.norm_names.append(concept.names)

                gold_entity = gold_document.entities[entity_start + batch_idx]
                if len(gold_entity.norm_ids) == 0:
                    # if gold_entity not annotated, we count it as TP
                    pass
                else:

                    if len(predict_entity.norm_ids) != 0 and predict_entity.norm_ids[0] in UMLS_dict:
                        concept = UMLS_dict[predict_entity.norm_ids[0]]

                        if gold_entity.norm_ids[0] in concept.codes:

                            # correct
                            logging.info("correct | mention: {} | gold entity: {} | pred entity: {}".format(gold_entity.name, gold_entity.norm_names[0],
                                                                                              predict_entity.norm_names[
                                                                                                  0]))
                            encoded_layers = encoded_layers[:, 1:-1, :]
                            lengths = lengths - 2
                            attention_weight = attention_model.forward(encoded_layers, lengths)

                            tokens = []
                            for token in tokenizer.tokenize(gold_entity.name):
                                if token in stop_word:
                                    continue
                                token = word_preprocess(token)
                                tokens.append(token)

                            # logging.info(attention_weight)
                            # make_heatmaps("model1", tokens, attention_weight[0].detach().numpy(), "GnBu")
                        else:
                            # wrong
                            logging.info("wrong | mention: {} | gold entity: {} | pred entity: {}".format(gold_entity.name, gold_entity.norm_names[0],
                                                                                              predict_entity.norm_names[
                                                                                                  0]))
                            encoded_layers = encoded_layers[:, 1:-1, :]
                            lengths = lengths - 2
                            attention_weight = attention_model.forward(encoded_layers, lengths)

                            tokens = []
                            for token in tokenizer.tokenize(gold_entity.name):
                                if token in stop_word:
                                    continue
                                token = word_preprocess(token)
                                tokens.append(token)

                            # logging.info(attention_weight)
                            make_heatmaps("model1", tokens, attention_weight[0].detach().numpy(), "GnBu")

            entity_start += actual_batch_size


        break





