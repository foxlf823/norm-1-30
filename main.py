
import random
import numpy as np
import torch
import os
import logging

from options import opt
import data
import train
import test
from my_utils import makedir_and_clear
import umls
import norm_neural
import vsm
import ensemble
import multi_sieve

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

if opt.whattodo == 1:
    logging.info(d.config)

    logging.info("load data ...")
    d.train_data = data.loadData(opt.train_file, True, opt.types, opt.type_filter)
    d.dev_data = data.loadData(opt.dev_file, True, opt.types, opt.type_filter)
    if opt.test_file:
        d.test_data = data.loadData(opt.test_file, False, opt.types, opt.type_filter)

    logging.info("build alphabet ...")
    d.build_alphabet(d.train_data)
    d.build_alphabet(d.dev_data)
    if opt.test_file:
        d.build_alphabet(d.test_data)

    d.fix_alphabet()

    logging.info("generate instance ...")
    d.train_texts, d.train_Ids = data.read_instance(d.train_data, d.word_alphabet, d.char_alphabet, d.label_alphabet, d)
    d.dev_texts, d.dev_Ids = data.read_instance(d.dev_data, d.word_alphabet, d.char_alphabet, d.label_alphabet, d)
    if opt.test_file:
        d.test_texts, d.test_Ids = data.read_instance(d.test_data, d.word_alphabet, d.char_alphabet, d.label_alphabet, d)

    logging.info("load pretrained word embedding ...")
    d.pretrain_word_embedding, d.word_emb_dim = data.build_pretrain_embedding(opt.word_emb_file, d.word_alphabet, opt.word_emb_dim, False)

    makedir_and_clear(opt.output)

    train.train(d, opt, None)

    d.clear() # clear some data due it's useless when test
    d.save(os.path.join(opt.output, "data.pkl"))

elif opt.whattodo == 2:
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


    if opt.norm_rule and opt.norm_vsm and opt.norm_neural:  # ensemble
        ensemble.train(train_data, dev_data, test_data, d, UMLS_dict, UMLS_dict_reverse, opt, None, False)
    elif opt.norm_rule:
        p, r, f = multi_sieve.train(train_data, dev_data, d, UMLS_dict, UMLS_dict_reverse, opt, None, False)
    elif opt.norm_vsm:
        vsm.train(train_data, dev_data, test_data, d, UMLS_dict, UMLS_dict_reverse, opt, None, False)
    elif opt.norm_neural:
        norm_neural.train(train_data, dev_data, test_data, d, UMLS_dict, UMLS_dict_reverse, opt, None, False)
    else:
        raise RuntimeError("wrong configuration")

else:
    d.load(os.path.join(opt.output, "data.pkl"))

    d_new = data.Data(opt)
    d.config = d_new.config
    logging.info(d.config)

    test.test(d, opt)


