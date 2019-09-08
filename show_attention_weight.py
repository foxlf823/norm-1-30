import matplotlib.pyplot as plt
import numpy as np
import logging
from options import opt
import data
import umls
import os
import torch

from my_utils import get_bioc_file, get_text_file
import metamap
from data_structure import Document, Entity
from pytorch_pretrained_bert import BertTokenizer, BertModel
from pytorch_pretrained_bert.modeling import BertPreTrainedModel
import torch.nn as nn

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


def get_dict_size(dict_alphabet):
    return dict_alphabet.size()-2

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

        Xs, Ys = generate_instances_ehr(entities, self.dict_alphabet, dictionary_reverse)

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

                concept = dictionary[norm_id]
                entity.norm_ids.append(norm_id)
                entity.norm_names.append(concept.names)

            entity_start += actual_batch_size

def evaluate_for_ehr(gold_entities, pred_entities, dictionary):

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

                break

    return ct_norm_gold, ct_norm_predict, ct_norm_correct

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
    if(words_or_sent):
        fig.set_size_inches(fig_size_v, 2)
    else:
        fig.set_size_inches(7, 2)
    plt.show()




if __name__ == '__main__':

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

    for gold_file_name in annotation_files:
        print("# begin {}".format(gold_file_name))
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

        model.process_one_doc(gold_document, pred_entities, UMLS_dict, UMLS_dict_reverse)

        p1, p2, p3 = evaluate_for_ehr(gold_document.entities, pred_entities, UMLS_dict)

        break

    # make_heatmaps("model1", ["a", "b", "c"], [0.1, 0.2, 0.7], "GnBu")