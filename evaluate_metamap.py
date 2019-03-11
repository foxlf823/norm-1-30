import umls
import metamap
from os import listdir
from os.path import isfile, join
import os
from my_utils import get_bioc_file
from data_structure import Entity,Document
import logging
from norm_utils import evaluate_for_ehr
from options import opt
import data
import multi_sieve
from my_utils import get_text_file
import torch

type_we_care = set(['ADE','SSLIF', 'Indication'])

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def parse_one_gold_file(annotation_dir, corpus_dir, fileName):
    document = Document()
    document.name = fileName[:fileName.find('.')]

    annotation_file = get_bioc_file(join(annotation_dir, fileName))
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

    corpus_file = get_text_file(join(corpus_dir, fileName.split('.bioc')[0]))
    document.text = corpus_file

    return document


def metamap_ner_re(d):
    print("load umls ...")
    UMLS_dict, _ = umls.load_umls_MRCONSO(d.config['norm_dict'])


    predict_dir = "/Users/feili/Desktop/umass/CancerADE_SnoM_30Oct2017_test/metamap"
    annotation_dir = os.path.join(opt.test_file, 'bioc')
    corpus_dir = os.path.join(opt.test_file, 'txt')

    annotation_files = [f for f in listdir(annotation_dir) if isfile(join(annotation_dir, f))]

    ct_ner_predict = 0
    ct_ner_gold = 0
    ct_ner_correct = 0

    ct_norm_predict = 0
    ct_norm_gold = 0
    ct_norm_correct = 0


    for gold_file_name in annotation_files:

        gold_document = parse_one_gold_file(annotation_dir, corpus_dir, gold_file_name)

        predict_document = metamap.load_metamap_result_from_file(join(predict_dir, gold_file_name[:gold_file_name.find('.')]+".field.txt"))

        ct_ner_gold += len(gold_document.entities)
        ct_ner_predict += len(predict_document.entities)

        for predict_entity in predict_document.entities:

            for gold_entity in gold_document.entities:

                if predict_entity.equals_span(gold_entity):

                    ct_ner_correct += 1

                    break



        p1, p2, p3 = evaluate_for_ehr(gold_document.entities, predict_document.entities, UMLS_dict)

        ct_norm_gold += p1
        ct_norm_predict += p2
        ct_norm_correct += p3


    p = ct_ner_correct*1.0/ct_ner_predict
    r = ct_ner_correct*1.0/ct_ner_gold
    f1 = 2.0*p*r/(p+r)
    print("NER p: %.4f | r: %.4f | f1: %.4f" % (p, r, f1))

    p = ct_norm_correct*1.0/ct_norm_predict
    r = ct_norm_correct*1.0/ct_norm_gold
    f1 = 2.0*p*r/(p+r)
    print("NORM p: %.4f | r: %.4f | f1: %.4f" % (p, r, f1))


def metamap_ner_my_norm(d):
    print("load umls ...")

    UMLS_dict, UMLS_dict_reverse = umls.load_umls_MRCONSO(d.config['norm_dict'])

    predict_dir = "/Users/feili/Desktop/umass/CancerADE_SnoM_30Oct2017_test/metamap"
    annotation_dir = os.path.join(opt.test_file, 'bioc')
    corpus_dir = os.path.join(opt.test_file, 'txt')
    annotation_files = [f for f in listdir(annotation_dir) if isfile(join(annotation_dir, f))]

    if opt.norm_rule:
        multi_sieve.init(opt, None, d, UMLS_dict, UMLS_dict_reverse, False)
    elif opt.norm_neural:
        logging.info("use neural-based normer")
        if opt.test_in_cpu:
            neural_model = torch.load(os.path.join(opt.output, 'norm_neural.pkl'), map_location='cpu')
        else:
            neural_model = torch.load(os.path.join(opt.output, 'norm_neural.pkl'))
        neural_model.eval()
    elif opt.norm_vsm:
        logging.info("use vsm-based normer")
        if opt.test_in_cpu:
            vsm_model = torch.load(os.path.join(opt.output, 'vsm.pkl'), map_location='cpu')
        else:
            vsm_model = torch.load(os.path.join(opt.output, 'vsm.pkl'))
        vsm_model.eval()

    ct_norm_predict = 0
    ct_norm_gold = 0
    ct_norm_correct = 0

    for gold_file_name in annotation_files:
        print("# begin {}".format(gold_file_name))
        gold_document = parse_one_gold_file(annotation_dir, corpus_dir, gold_file_name)

        predict_document = metamap.load_metamap_result_from_file(
            join(predict_dir, gold_file_name[:gold_file_name.find('.')] + ".field.txt"))

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

        if opt.norm_rule:
            multi_sieve.runMultiPassSieve(gold_document, pred_entities, UMLS_dict, False)
        elif opt.norm_neural:
            neural_model.process_one_doc(gold_document, pred_entities, UMLS_dict, UMLS_dict_reverse, False)
        elif opt.norm_vsm:
            vsm_model.process_one_doc(gold_document, pred_entities, UMLS_dict, UMLS_dict_reverse, False)
        else:
            raise RuntimeError("wrong configuration")

        p1, p2, p3 = evaluate_for_ehr(gold_document.entities, pred_entities, UMLS_dict)

        ct_norm_gold += p1
        ct_norm_predict += p2
        ct_norm_correct += p3


    p = ct_norm_correct*1.0/ct_norm_predict
    r = ct_norm_correct*1.0/ct_norm_gold
    f1 = 2.0*p*r/(p+r)
    print("NORM p: %.4f | r: %.4f | f1: %.4f" % (p, r, f1))


if __name__=="__main__":

    d = data.Data(opt)

    # metamap_ner_re(d)
    metamap_ner_my_norm(d)

