from seqmodel import SeqModel
import torch
import os
from my_utils import evaluate, makedir_and_clear
import codecs
from os.path import isfile, join
from os import listdir
# import spacy
from data import processOneFile, read_instance_from_one_document
import bioc
from data_structure import Entity
import logging
import nltk
from my_corenlp_wrapper import StanfordCoreNLP
import time
import umls
import multi_sieve
import copy
import ensemble
from metric import get_ner_BMES



def translateResultsintoEntities(sentences, predict_results):

    pred_entities = []
    sent_num = len(predict_results)

    for idx in range(sent_num):

        predict_list = predict_results[idx]
        sentence = sentences[idx]

        entities = get_ner_BMES(predict_list, False)

        # find span based on tkSpan, fill name
        for entity in entities:
            name = ''
            for tkSpan in entity.tkSpans:
                span = [sentence[tkSpan[0]]['start'], sentence[tkSpan[1]]['end']]
                entity.spans.append(span)
                for i in range(tkSpan[0], tkSpan[1]+1):
                    name += sentence[i]['text'] + ' '
            entity.name = name.strip()

        pred_entities.extend(entities)


    return pred_entities


def dump_results(doc_name, entities, opt):
    entity_id = 1
    collection = bioc.BioCCollection()
    document = bioc.BioCDocument()
    collection.add_document(document)
    document.id = doc_name
    passage = bioc.BioCPassage()
    document.add_passage(passage)
    passage.offset = 0

    for entity in entities:
        anno_entity = bioc.BioCAnnotation()
        passage.add_annotation(anno_entity)
        anno_entity.id = str(entity_id)
        entity_id += 1
        anno_entity.infons['type'] = entity.type
        anno_entity_location = bioc.BioCLocation(entity.spans[0][0], entity.spans[0][1]-entity.spans[0][0])
        anno_entity.add_location(anno_entity_location)
        anno_entity.text = entity.name
        if len(entity.norm_ids) > 0:
            anno_entity.infons['UMLS code'] = entity.norm_ids[0]
            anno_entity.infons['UMLS term'] = entity.norm_names[0]
        else:
            anno_entity.infons['UMLS code'] = 'N/A'
            anno_entity.infons['UMLS term'] = 'N/A'

    with codecs.open(os.path.join(opt.predict, doc_name + ".bioc.xml"), 'w', 'UTF-8') as fp:
        bioc.dump(collection, fp)



def test(data, opt):
    # corpus_dir = join(opt.test_file, 'corpus')
    # corpus_dir = join(opt.test_file, 'txt')
    corpus_dir = opt.test_file

    if opt.nlp_tool == "spacy":
        nlp_tool = spacy.load('en')
    elif opt.nlp_tool == "nltk":
        nlp_tool = nltk.data.load('tokenizers/punkt/english.pickle')
    elif opt.nlp_tool == "stanford":
        nlp_tool = StanfordCoreNLP('http://localhost:{0}'.format(9000))
    else:
        raise RuntimeError("invalid nlp tool")

    corpus_files = [f for f in listdir(corpus_dir) if isfile(join(corpus_dir, f))]

    model = SeqModel(data, opt)
    if opt.test_in_cpu:
        model.load_state_dict(
            torch.load(os.path.join(opt.output, 'model.pkl'), map_location='cpu'))
    else:
        model.load_state_dict(torch.load(os.path.join(opt.output, 'model.pkl')))

    dictionary, dictionary_reverse = umls.load_umls_MRCONSO(data.config['norm_dict'])
    isMeddra_dict = False

    # initialize norm models
    if opt.norm_rule and opt.norm_vsm and opt.norm_neural: # ensemble
        logging.info("use ensemble normer")
        multi_sieve.init(opt, None, data, dictionary, dictionary_reverse, False)
        if opt.ensemble == 'learn':
            if opt.test_in_cpu:
                ensemble_model = torch.load(os.path.join(opt.output, 'ensemble.pkl'), map_location='cpu')
            else:
                ensemble_model = torch.load(os.path.join(opt.output, 'ensemble.pkl'))
            ensemble_model.eval()
        else:
            if opt.test_in_cpu:
                vsm_model = torch.load(os.path.join(opt.output, 'vsm.pkl'), map_location='cpu')
                neural_model = torch.load(os.path.join(opt.output, 'norm_neural.pkl'), map_location='cpu')
            else:
                vsm_model = torch.load(os.path.join(opt.output, 'vsm.pkl'))
                neural_model = torch.load(os.path.join(opt.output, 'norm_neural.pkl'))

            vsm_model.eval()
            neural_model.eval()

    elif opt.norm_rule:
        logging.info("use rule-based normer")
        multi_sieve.init(opt, None, data, dictionary, dictionary_reverse, False)

    elif opt.norm_vsm:
        logging.info("use vsm-based normer")
        if opt.test_in_cpu:
            vsm_model = torch.load(os.path.join(opt.output, 'vsm.pkl'), map_location='cpu')
        else:
            vsm_model = torch.load(os.path.join(opt.output, 'vsm.pkl'))
        vsm_model.eval()

    elif opt.norm_neural:
        logging.info("use neural-based normer")
        if opt.test_in_cpu:
            neural_model = torch.load(os.path.join(opt.output, 'norm_neural.pkl'), map_location='cpu')
        else:
            neural_model = torch.load(os.path.join(opt.output, 'norm_neural.pkl'))
        neural_model.eval()
    else:
        logging.info("no normalization is performed.")

    makedir_and_clear(opt.predict)

    ct_success = 0
    ct_error = 0

    for fileName in corpus_files:
        try:
            start = time.time()
            document, _, _, _ = processOneFile(fileName, None, corpus_dir, nlp_tool, False, opt.types, opt.type_filter)

            data.test_texts = []
            data.test_Ids = []
            read_instance_from_one_document(document, data.word_alphabet, data.char_alphabet, data.label_alphabet,
                                            data.test_texts, data.test_Ids, data)

            _, _, _, _, _, pred_results, _ = evaluate(data, opt, model, 'test', False, opt.nbest)

            entities = translateResultsintoEntities(document.sentences, pred_results)

            if opt.norm_rule and opt.norm_vsm and opt.norm_neural:
                if opt.ensemble == 'learn':
                    ensemble_model.process_one_doc(document, entities, dictionary, dictionary_reverse, isMeddra_dict)
                else:
                    pred_entities1 = copy.deepcopy(entities)
                    pred_entities2 = copy.deepcopy(entities)
                    pred_entities3 = copy.deepcopy(entities)
                    multi_sieve.runMultiPassSieve(document, pred_entities1, dictionary, isMeddra_dict)
                    vsm_model.process_one_doc(document, pred_entities2, dictionary, dictionary_reverse, isMeddra_dict)
                    neural_model.process_one_doc(document, pred_entities3, dictionary, dictionary_reverse, isMeddra_dict)

                    # merge pred_entities1, pred_entities2, pred_entities3 into entities
                    ensemble.merge_result(pred_entities1, pred_entities2, pred_entities3, entities, dictionary, isMeddra_dict,
                                          vsm_model.dict_alphabet, data)

            elif opt.norm_rule:
                multi_sieve.runMultiPassSieve(document, entities, dictionary, isMeddra_dict)
            elif opt.norm_vsm:
                vsm_model.process_one_doc(document, entities, dictionary, dictionary_reverse, isMeddra_dict)
            elif opt.norm_neural:
                neural_model.process_one_doc(document, entities, dictionary, dictionary_reverse, isMeddra_dict)


            dump_results(fileName, entities, opt)

            end = time.time()
            logging.info("process %s complete with %.2fs" % (fileName, end - start))

            ct_success += 1
        except Exception as e:
            logging.error("process file {} error: {}".format(fileName, e))
            ct_error += 1

    logging.info("test finished, total {}, error {}".format(ct_success + ct_error, ct_error))