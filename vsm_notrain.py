import torch.nn as nn
import logging
from alphabet import Alphabet
from my_utils import random_embedding
import torch
from data import build_pretrain_embedding, my_tokenize, load_data_fda
import numpy as np
import torch.nn.functional as functional
import os
from data_structure import Entity
import norm_utils
from options import opt

class VsmNormer(nn.Module):

    def __init__(self):
        super(VsmNormer, self).__init__()
        self.word_alphabet = Alphabet('word')
        self.embedding_dim = None
        self.word_embedding = None
        self.dict_alphabet = Alphabet('dict')
        self.dict_embedding = None
        self.gpu = opt.gpu

    def transfer_model_into_gpu(self):
        if torch.cuda.is_available():
            self.word_embedding = self.word_embedding.cuda(self.gpu)
            self.dict_embedding = self.dict_embedding.cuda(self.gpu)


    def batch_name_to_ids(self, name):
        tokens = my_tokenize(name)
        length = len(tokens)
        tokens_id = np.zeros((1, length), dtype=np.int)
        for i, word in enumerate(tokens):
            word = norm_utils.word_preprocess(word)
            tokens_id[0][i] = self.word_alphabet.get_index(word)

        tokens_id = torch.from_numpy(tokens_id)

        if torch.cuda.is_available():
            return tokens_id.cuda(self.gpu)
        else:
            return tokens_id


    def init_vector_for_dict(self, meddra_dict):
        self.dict_embedding = nn.Embedding(len(meddra_dict), self.embedding_dim)
        if torch.cuda.is_available():
            self.dict_embedding = self.dict_embedding.cuda(self.gpu)

        for concept_id, concept_name in meddra_dict.items():
            self.dict_alphabet.add(concept_id)
            with torch.no_grad():
                tokens_id = self.batch_name_to_ids(concept_name)
                length = tokens_id.size(1)
                emb = self.word_embedding(tokens_id)
                emb = emb.unsqueeze_(1)
                pool = functional.avg_pool2d(emb, (length, 1))
                index = norm_utils.get_dict_index(self.dict_alphabet, concept_id)
                self.dict_embedding.weight.data[index] = pool[0][0]

    def compute_similarity(self, mention_rep, concep_rep):
        # mention_rep is (batch, emb_dim) and concep_rep is (concept_num, emb_dim)
        mention_rep_norm = torch.norm(mention_rep, 2, 1, True)  # batch 1
        concep_rep_norm = torch.norm(concep_rep, 2, 1, True)  # concept 1
        a = torch.matmul(mention_rep_norm, torch.t(concep_rep_norm)) # batch, concept
        a = a.clamp(min=1e-8)

        b = torch.matmul(mention_rep, torch.t(concep_rep)) # batch, concept

        return b / a


    def forward(self, mention_word_ids):
        length = mention_word_ids.size(1)
        mention_word_emb = self.word_embedding(mention_word_ids)
        mention_word_emb = mention_word_emb.unsqueeze_(1)
        mention_word_pool = functional.avg_pool2d(mention_word_emb, (length, 1)) # batch,1,1,100
        mention_word_pool = mention_word_pool.squeeze_(1).squeeze_(1) # batch,100

        # similarities = torch.t(torch.matmul(self.dict_embedding.weight.data, torch.t(mention_word_pool))) # batch, dict
        similarities = self.compute_similarity(mention_word_pool, self.dict_embedding.weight.data)

        values, indices = torch.max(similarities, 1)

        return values, indices

    def process_one_doc(self, doc, entities, dict):

        for entity in entities:
            with torch.no_grad():
                tokens_id = self.batch_name_to_ids(entity.name)

                values, indices = self.forward(tokens_id)

                norm_id = norm_utils.get_dict_name(self.dict_alphabet, indices.item())
                name = dict[norm_id]
                entity.norm_ids.append(norm_id)
                entity.norm_names.append(name)
                entity.norm_confidences.append(values.item())




def train(train_data, dev_data, d, meddra_dict, opt, fold_idx):
    logging.info("train the vsm-based normalization model ...")

    external_train_data = []
    if d.config.get('norm_ext_corpus') is not None:
        for k, v in d.config['norm_ext_corpus'].items():
            if k == 'tac':
                external_train_data.extend(load_data_fda(v['path'], True, v.get('types'), v.get('types'), False, True))
            else:
                raise RuntimeError("not support external corpus")
    if len(external_train_data) != 0:
        train_data.extend(external_train_data)

    vsm_model = VsmNormer()

    logging.info("build alphabet ...")
    norm_utils.build_alphabet(vsm_model.word_alphabet, train_data)
    if opt.dev_file:
        norm_utils.build_alphabet(vsm_model.word_alphabet, dev_data)

    norm_utils.build_alphabet_from_dict(vsm_model.word_alphabet, meddra_dict)
    norm_utils.fix_alphabet(vsm_model.word_alphabet)

    if d.config.get('norm_emb') is not None:
        logging.info("load pretrained word embedding ...")
        pretrain_word_embedding, word_emb_dim = build_pretrain_embedding(d.config.get('norm_emb'),
                                                                              vsm_model.word_alphabet,
                                                                              opt.word_emb_dim, False)
        vsm_model.word_embedding = nn.Embedding(vsm_model.word_alphabet.size(), word_emb_dim)
        vsm_model.word_embedding.weight.data.copy_(torch.from_numpy(pretrain_word_embedding))
        vsm_model.embedding_dim = word_emb_dim
    else:
        logging.info("randomly initialize word embedding ...")
        vsm_model.word_embedding = nn.Embedding(vsm_model.word_alphabet.size(), d.word_emb_dim)
        vsm_model.word_embedding.weight.data.copy_(
            torch.from_numpy(random_embedding(vsm_model.word_alphabet.size(), d.word_emb_dim)))
        vsm_model.embedding_dim = d.word_emb_dim

    if torch.cuda.is_available():
        vsm_model.word_embedding = vsm_model.word_embedding.cuda(vsm_model.gpu)

    logging.info("init_vector_for_dict")
    vsm_model.init_vector_for_dict(meddra_dict)
    norm_utils.fix_alphabet(vsm_model.dict_alphabet)

    vsm_model.train()

    best_dev_f = -10
    best_dev_p = -10
    best_dev_r = -10

    if opt.dev_file:
        p, r, f = norm_utils.evaluate(dev_data, meddra_dict, vsm_model)
        logging.info("Dev: p: %.4f, r: %.4f, f: %.4f" % (p, r, f))
    else:
        f = best_dev_f

    if f > best_dev_f:
        logging.info("Exceed previous best f score on dev: %.4f" % (best_dev_f))

        if fold_idx is None:
            logging.info("save model to {}".format(os.path.join(opt.output, "vsm.pkl")))
            torch.save(vsm_model, os.path.join(opt.output, "vsm.pkl"))
        else:
            logging.info("save model to {}".format(os.path.join(opt.output, "vsm_{}.pkl".format(fold_idx + 1))))
            torch.save(vsm_model, os.path.join(opt.output, "vsm_{}.pkl".format(fold_idx + 1)))

        best_dev_f = f
        best_dev_p = p
        best_dev_r = r


    logging.info("train finished")

    if len(opt.dev_file) == 0:
        logging.info("save model to {}".format(os.path.join(opt.output, "vsm.pkl")))
        torch.save(vsm_model, os.path.join(opt.output, "vsm.pkl"))

    return best_dev_p, best_dev_r, best_dev_f