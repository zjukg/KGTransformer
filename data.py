from ast import While
import imp
from multiprocessing.reduction import sendfds
from turtle import pd
from torch.utils.data import Dataset, DataLoader
import copy
from tqdm import tqdm
import torch
import random
from collections import defaultdict as ddict
import os
import pickle as pkl
import json
import numpy as np
import scipy.io as sio
from sklearn import preprocessing
import sys
import time
from qa_utils import load_input_tensors, MODEL_NAME_TO_CLASS

def timer(func):
    def wrapper_triple(*args, **kwargs):
        begin = time.time()
        value = func(*args, **kwargs)
        end = time.time()
        print(f"timecout {func.__name__!r} {end-begin}s")
        return value
    return wrapper_triple

def return_deafule_cls():
    return -1

class KGTokenizer:
    
    def __init__(self,args) -> None:
        """Tokenizer of kg data.

        Attributes:
            args: Some pre-set parameters, such as dataset path, etc.
        """
        self.args = args

        self.ent2id = {}
        self.rel2id = {}
        self.token2id = {}
        # predictor
        self.id2ent = {}
        self.id2rel = {}
        self.id2token = {}
        # triple id
        self.train_triples = []
        self.valid_triples = []
        self.test_triples = []
        self.all_true_triples = set()

        print('get_entity_id')
        self.get_id()
        
        
        print('get_triples_id')
        if 'qa' in self.args.down_task:
            print('qa_task pass, do not need triple id.')
        else:
            self.get_triples_id() 
            print('get_entity_related_triples')
            self.get_entity_related_triples() 
        

        if not args.debug:
            if args.down_task:
                print('downstream task, pass')
            else:
                print('get_2_hop_related_triples')
                self.get_2_hop_related_triples()
        else:
            print('debug, pass')

        
        self.rel_list = list(self.id2rel.keys())

        
    def get_id(self):
        """Get entity/relation id, and entity/relation number.

        Update:
            self.ent2id: Entity to id.
            self.rel2id: Relation to id.
            self.id2ent: id to Entity.
            self.id2rel: id to Relation.
            self.args.num_ent: Entity number.
            self.args.num_rel: Relation number.
        """

        for index,token in enumerate(['[CLS]','[SEP]','[MASK]','[TASK]','[ENT1]','[ENT2]','[PIC]','[PAD]','[SEP_G]']+[f'[unused{i}]' for i in range(100)]):
            self.token2id[token] = index
            self.id2token[index] = token

        with open(os.path.join(self.args.data_path, "entities.dict")) as fin:
            for line in fin:
                eid, entity = line.strip().split("\t")
                self.ent2id[entity] = int(eid)+len(self.token2id)
                self.id2ent[int(eid)+len(self.token2id)] = entity

        with open(os.path.join(self.args.data_path, "relations.dict")) as fin:
            for line in fin:
                rid, relation = line.strip().split("\t")
                self.rel2id[relation] = int(rid)+len(self.ent2id)+len(self.token2id)
                self.id2rel[int(rid)+len(self.ent2id)+len(self.token2id)] = relation

        print(list(self.rel2id.values())[-1])
        # assert list(self.rel2id.values())[-1] == len(self.ent2id) + len(self.token2id) +len(self.rel2id) - 1

        
        self.args.num_ent = len(self.ent2id)
        self.args.num_rel = len(self.rel2id)
        self.args.num_token = len(self.token2id)


    def down_data_linkpre(self):
        self.down_data_train = []
        self.down_data_valid = []
        self.down_data_test = []

        self.entity_subgraph_fixed_h = {}
        self.entity_subgraph_fixed_t = {}

        with open(os.path.join(self.args.data_path, "train.txt")) as f:
            for line in f.readlines():
                h, r, t = line.strip().split()
                self.down_data_train.append(
                    (self.ent2id[h], self.rel2id[r], self.ent2id[t])
                )

        with open(os.path.join(self.args.data_path, "valid.txt")) as f:
            for line in f.readlines():
                h, r, t = line.strip().split()
                self.down_data_valid.append(
                    (self.ent2id[h], self.rel2id[r], self.ent2id[t])
                )

        with open(os.path.join(self.args.data_path, "test.txt")) as f:
            for line in f.readlines():
                h, r, t = line.strip().split()
                self.down_data_test.append(
                    (self.ent2id[h], self.rel2id[r], self.ent2id[t])
                )

        self.get_hr2t_rt2h_for_filter()
        self.get_hr2t_rt2h_in_train()
    
    def down_data_entsub(self):
        self.down_data_train = []
        self.down_data_valid = []
        self.down_data_test = []
        self.true_train_pair = []
        self.true_valid_pair = []
        self.true_test_pair = []
        self.candidat_entities = set()

        with open(os.path.join(self.args.data_path, self.args.down_task, "train.txt")) as f:
            for line in f.readlines():
                h, t = line.strip().split()
                hid = self.ent2id[h]
                tid = self.ent2id[t]
                self.down_data_train.append((hid, tid))
                self.true_train_pair.append((hid, tid))
                self.candidat_entities.add(hid)
                self.candidat_entities.add(tid)

        with open(os.path.join(self.args.data_path, self.args.down_task, "valid.txt")) as f:
            for line in f.readlines():
                h, t, label = line.strip().split()
                hid = self.ent2id[h]
                tid = self.ent2id[t]
                self.candidat_entities.add(hid)
                self.candidat_entities.add(tid)
                self.down_data_valid.append((hid, tid, int(label)))
                if label == '1':
                    self.true_valid_pair.append((hid, tid))

        with open(os.path.join(self.args.data_path, self.args.down_task, "test.txt")) as f:
            for line in f.readlines():
                h, t, label = line.strip().split()
                hid = self.ent2id[h]
                tid = self.ent2id[t]
                self.candidat_entities.add(hid)
                self.candidat_entities.add(tid)
                self.down_data_test.append((hid, tid, int(label)))
                if label == '1':
                    self.true_test_pair.append((hid, tid))

        self.all_true_down_samples = set(self.true_train_pair + self.true_valid_pair + self.true_test_pair)
        self.candidat_entities_list = list(self.candidat_entities)

    def down_data_triplecls(self):
        self.down_data_train = []
        self.down_data_valid = []
        self.down_data_test = []
        self.candi_entity_ids = set()
        self.triple_false_in_test = set()

        # -----------------------
        # subgraph of entity
        # -----------------------
        self.entity_subgraph_fixed = {}

        with open(os.path.join(self.args.data_path, self.args.down_task, "train.txt")) as f:
            for line in f.readlines():
                h, r, t = line.strip().split()
                hid = self.ent2id[h]
                rid = self.rel2id[r]
                tid = self.ent2id[t]
                self.candi_entity_ids.add(hid)
                self.candi_entity_ids.add(tid)
                self.down_data_train.append((hid, rid, tid))
                
        with open(os.path.join(self.args.data_path, self.args.down_task, "valid.txt")) as f:
            for line in f.readlines():
                h, r, t, label = line.strip().split()
                hid = self.ent2id[h]
                rid = self.rel2id[r]
                tid = self.ent2id[t]
                self.candi_entity_ids.add(hid)
                self.candi_entity_ids.add(tid)
                self.down_data_valid.append((hid, rid, tid, int(label)))
                if label != '1':
                    self.triple_false_in_test.add((hid, rid, tid))

        with open(os.path.join(self.args.data_path, self.args.down_task, "test.txt")) as f:
            for line in f.readlines():
                h, r, t, label = line.strip().split()
                hid = self.ent2id[h]
                rid = self.rel2id[r]
                tid = self.ent2id[t]
                self.candi_entity_ids.add(hid)
                self.candi_entity_ids.add(tid)
                self.down_data_test.append((hid, rid, tid, int(label)))
                if label != '1':
                    self.triple_false_in_test.add((hid, rid, tid))

        self.get_2_hop_related_triples() 
        

    def down_data_zsl(self):
        # embedding and label of resnet
        matcontent = sio.loadmat("0419_KGPretrain_zsl/step1_featuredeal/res101.mat")
        feature = matcontent['features'].T # # 37322 2048
        label = matcontent['labels'].astype(int).squeeze() - 1  # 37322

        matcontent = sio.loadmat("0419_KGPretrain_zsl/step1_featuredeal/att_splits.mat")
        attribute = matcontent['att']
        attribute_kg = attribute #np.load("dataset/down_zsl/attri_mat.npy")
        trainval_loc = matcontent['trainval_loc'].squeeze() - 1   # 23527
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1   # 5882
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1   # 7913


        preprocessing_ornot = True
        standardization = True

        if preprocessing_ornot:
            if standardization:
                print('standardization...')
                scaler = preprocessing.StandardScaler()
            else:
                scaler = preprocessing.MinMaxScaler()

            _train_feature = scaler.fit_transform(feature[trainval_loc])
            _test_seen_feature = scaler.transform(feature[test_seen_loc])
            _test_unseen_feature = scaler.transform(feature[test_unseen_loc])

            train_feature = torch.from_numpy(_train_feature).float()
            mx = train_feature.max()
            train_feature.mul_(1/mx) # [23527, 2048]
            train_label = torch.from_numpy(label[trainval_loc]).long() # [23527]

            # import pdb; pdb.set_trace()
            rand_idx = np.arange(len(train_label))
            np.random.shuffle(rand_idx)

            train_feature = train_feature[rand_idx]
            train_label = train_label[rand_idx]

            test_unseen_feature = torch.from_numpy(_test_unseen_feature).float() # [7913, 2048]
            test_unseen_feature.mul_(1/mx)
            test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long()

            test_seen_feature = torch.from_numpy(_test_seen_feature).float() # [5882, 2048]
            test_seen_feature.mul_(1/mx) # [5882, 2048] 
            test_seen_label = torch.from_numpy(label[test_seen_loc]).long()
        else:
            train_feature = torch.from_numpy(feature[trainval_loc]).float()
            train_label = torch.from_numpy(label[trainval_loc]).long()

            # import pdb; pdb.set_trace()
            rand_idx = np.arange(len(train_label))
            np.random.shuffle(rand_idx)
            train_feature = train_feature[rand_idx]
            train_label = train_label[rand_idx]


            test_unseen_feature = torch.from_numpy(feature[test_unseen_loc]).float()
            test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long()

            test_seen_feature = torch.from_numpy(feature[test_seen_loc]).float()
            test_seen_label = torch.from_numpy(label[test_seen_loc]).long()

        
        self.seenclasses = sorted(np.unique(test_seen_label.numpy()), reverse = False)
        self.unseenclasses = sorted(np.unique(test_unseen_label.numpy()), reverse = False) 
        
        self.unseenclass_set = set(self.unseenclasses)
        self.seenclass_set = set(self.seenclasses)
        
        self.entity_subgraph_fixed = {}

        # import pdb; pdb.set_trace()
        # -----------------------
        # class and entity name, label
        # -----------------------
        with open ("0419_KGPretrain_zsl/step2_mapping/awa2_classid2name.json","r") as f:
            self.awa2_classid2name = json.load(f)
            for key,value in self.awa2_classid2name.items():
                self.awa2_classid2name[key] = value.replace('+','_')
            # print(self.awa2_classid2name)
        
        with open('./dataset/down_zsl/awacls_entity.txt','r') as f:
            for line in tqdm(f):
                clsid, cls_name, entity = line.strip().split('\t')
                assert self.awa2_classid2name[clsid] == cls_name 
                self.awa2_classid2name[clsid] = entity 

        self.down_data_train = []
        self.down_data_test_seen = []
        self.down_data_test_unseen = []

        
        self.down_data_train_comb = [] #(cid, ((fid0, fcl0, label0), (fid1, fcl1, label1), (,,))) 
        self.down_data_test_seen_comb = []
        self.down_data_test_unseen_comb = []

        self.attribute = torch.tensor(attribute)
        self.attribute_kg = torch.tensor(attribute_kg.T)
        used_entities = set()
        self.cid_to_fcls = ddict(return_deafule_cls) 
        # import pdb; pdb.set_trace()
        for index, the_label in enumerate(train_label.numpy()):
            cname = self.awa2_classid2name[str(the_label)] 
            cid = self.ent2id[cname]
            fid = index
            self.down_data_train.append((cid, fid, the_label))
            used_entities.add(cid)
            self.cid_to_fcls[cid] = the_label # fcls

            if index%self.args.multi_pic==0:
                new_combine = [cid]
                self.down_data_train_comb.append(new_combine)
            if the_label == self.cid_to_fcls[new_combine[0]]:
                the_match = 1
            else:
                the_match = 0
            new_combine.append((fid, the_label, the_match))
        
        if len(self.down_data_train_comb[-1])<self.args.multi_pic + 1:
            self.down_data_train_comb[-1] += [self.down_data_train_comb[-1][-1]]*(self.args.multi_pic+1-len(self.down_data_train_comb[-1]))
        
        # assert set([i[0] for i in self.tokenizer.down_data_train_comb]) == 40
        # dic_mm = defaultdict(int)
        # for i in mm:
            # dic_mm[i] += 1

        for index, the_label in enumerate(test_seen_label.numpy()):
            cname = self.awa2_classid2name[str(the_label)]
            cid = self.ent2id[cname]
            fid = index + len(self.down_data_train)
            self.down_data_test_seen.append((cid, fid, the_label, 'test'))
            used_entities.add(cid)
            self.cid_to_fcls[cid] = the_label

            if index % self.args.multi_pic==0:
                new_combine = []
                self.down_data_test_seen_comb.append(new_combine)
            new_combine.append((fid, the_label, cid))

        if len(self.down_data_test_seen_comb[-1])<self.args.multi_pic:
            self.down_data_test_seen_comb[-1] += [self.down_data_test_seen_comb[-1][-1]]*(self.args.multi_pic-len(self.down_data_test_seen_comb[-1]))
        

        for index, the_label in enumerate(test_unseen_label.numpy()):
            cname = self.awa2_classid2name[str(the_label)]
            cid = self.ent2id[cname]
            fid = index + len(self.down_data_train) + len(self.down_data_test_seen)
            self.down_data_test_unseen.append((cid, fid, the_label, 'test'))
            used_entities.add(cid)
            self.cid_to_fcls[cid] = the_label

            if index % self.args.multi_pic==0:
                new_combine = []
                self.down_data_test_unseen_comb.append(new_combine)
            new_combine.append((fid, the_label, cid))

        
        if len(self.down_data_test_unseen_comb[-1])<self.args.multi_pic:
            self.down_data_test_unseen_comb[-1] += [self.down_data_test_unseen_comb[-1][-1]]*(self.args.multi_pic-len(self.down_data_test_unseen_comb[-1]))
        # import pdb; pdb.set_trace()


        self.features_tensor = torch.cat((train_feature, test_seen_feature, test_unseen_feature), dim = 0) # [37322, 2048]

        # print('get 2-hop neighbor for',sorted(used_entities))
        
        self.get_2_hop_related_triples(specific_entity = sorted(used_entities))
    

    def down_data_qa(self):

        train_statement_path = './qa_data/train.statement.jsonl'
        dev_statement_path = './qa_data/dev.statement.jsonl'
        test_statement_path = './qa_data/test.statement.jsonl'

        train_tri_list_path = './qa_data/train_tri_list.prune.pkl'
        dev_tri_list_path = './qa_data/dev_tri_list.prune.pkl'
        test_tri_list_path = './qa_data/test_tri_list.prune.pkl'

        with open('./qa_data/inhouse_split_qids.txt', 'r') as fin:
            inhouse_qids = set(line.strip() for line in fin)

        self.model_name = 'roberta-large'
        model_type = MODEL_NAME_TO_CLASS[self.model_name]
        max_seq_length = 100

        train_qids, train_labels, *train_encoder_data = load_input_tensors(train_statement_path,
                                                                                          model_type,
                                                                                          self.model_name,
                                                                                          max_seq_length)

        inhouse_train_indexes = [i for i, qid in enumerate(train_qids) if qid in inhouse_qids]
        inhouse_test_indexes = [i for i, qid in enumerate(train_qids) if qid not in inhouse_qids]

        self.train_qids = [train_qids[i] for i in inhouse_train_indexes]
        self.train_labels = [train_labels[i] for i in inhouse_train_indexes]
        self.train_encoder_data = [[d[i] for i in inhouse_train_indexes] for d in train_encoder_data]

        self.test_qids = [train_qids[i] for i in inhouse_test_indexes]
        self.test_labels = [train_labels[i] for i in inhouse_test_indexes]
        self.test_encoder_data = [[d[i] for i in inhouse_test_indexes] for d in train_encoder_data]

        self.dev_qids, self.dev_labels, *self.dev_encoder_data = load_input_tensors(dev_statement_path,
                                                                                    model_type,
                                                                                    self.model_name,
                                                                                    max_seq_length)

        # self.test_qids, self.test_labels, *self.test_encoder_data = load_input_tensors(test_statement_path,
        #                                                                                model_type,
        #                                                                                self.model_name,
        #                                                                                max_seq_length)

        self.num_choice = 5
        train_tri_list = pkl.load(open(train_tri_list_path, 'rb'))

        self.train_tri_list = []
        for i in inhouse_train_indexes:
            self.train_tri_list.extend(train_tri_list[i*self.num_choice: (i+1)*self.num_choice])

        self.test_tri_list = []
        for i in inhouse_test_indexes:
            self.test_tri_list.extend(train_tri_list[i*self.num_choice: (i+1)*self.num_choice])

        self.dev_tri_list = pkl.load(open(dev_tri_list_path, 'rb'))
        # self.test_tri_list = pkl.load(open(test_tri_list_path, 'rb'))

        # self.train_tri_list = [(np.array([[h + 109, r + 1101662, t + 109] for h, r, t in triples]), qamask)
        #                        for (triples, qamask) in self.train_tri_list]
        # self.dev_tri_list = [(np.array([[h + 109, r + 1101662, t + 109] for h, r, t in triples]), qamask)
        #                        for (triples, qamask) in self.dev_tri_list]
        # self.test_tri_list = [(np.array([[h + 109, r + 1101662, t + 109] for h, r, t in triples]), qamask)
        #                        for (triples, qamask) in self.test_tri_list]
        # import pdb; pdb.set_trace()
        
        self.train_tri_list = [(np.array([[self.ent2id[str(h)], self.rel2id[str(r)], self.ent2id[str(t)]] for h, r, t in triples[:]]), qamask[:]) 
                                for (triples, qamask) in self.train_tri_list]
        self.dev_tri_list = [(np.array([[self.ent2id[str(h)], self.rel2id[str(r)], self.ent2id[str(t)]] for h, r, t in triples[:]]), qamask[:])
                                for (triples, qamask) in self.dev_tri_list]
        self.test_tri_list = [(np.array([[self.ent2id[str(h)], self.rel2id[str(r)], self.ent2id[str(t)]] for h, r, t in triples[:]]), qamask[:])
                                for (triples, qamask) in self.test_tri_list]

        
    
    def get_hr2t_rt2h_for_filter(self):
        """Get the set of hr2t and rt2h from train, valid, test dataset, the data type is numpy.

        Update:
            self.hr2t: The set of hr2t.
            self.rt2h: The set of rt2h.
        """
        self.hr2t = ddict(set)
        self.rt2h = ddict(set)
        for h, r, t in self.train_triples + self.valid_triples + self.test_triples:
            self.hr2t[(h, r)].add(t)
            self.rt2h[(r, t)].add(h)
        for h, r in self.hr2t:
            self.hr2t[(h, r)] = np.array(list(self.hr2t[(h, r)]))
        for r, t in self.rt2h:
            self.rt2h[(r, t)] = np.array(list(self.rt2h[(r, t)]))
    
    def get_hr2t_rt2h_in_train(self):
        self.hr2t_train = ddict(set)
        self.rt2h_train = ddict(set)
        for h, r, t in self.train_triples:
            self.hr2t_train[(h, r)].add(t)
            self.rt2h_train[(r, t)].add(h)
        for h, r in self.hr2t_train:
            self.hr2t_train[(h, r)] = np.array(list(self.hr2t_train[(h, r)]))
        for r, t in self.rt2h_train:
            self.rt2h_train[(r, t)] = np.array(list(self.rt2h_train[(r, t)]))

    def get_triples_id(self):
        """Get triples id, save in the format of (h, r, t).

        Update:
            self.train_triples: Train dataset triples id.
            self.valid_triples: Valid dataset triples id.
            self.test_triples: Test dataset triples id.
        """
        
        with open(os.path.join(self.args.data_path, "train.txt")) as f:
            for line in f.readlines():
                h, r, t = line.strip().split()
                self.train_triples.append(
                    (self.ent2id[h], self.rel2id[r], self.ent2id[t])
                )

        with open(os.path.join(self.args.data_path, "valid.txt")) as f:
            for line in f.readlines():
                h, r, t = line.strip().split()
                self.valid_triples.append(
                    (self.ent2id[h], self.rel2id[r], self.ent2id[t])
                )

        with open(os.path.join(self.args.data_path, "test.txt")) as f:
            for line in f.readlines():
                h, r, t = line.strip().split()
                self.test_triples.append(
                    (self.ent2id[h], self.rel2id[r], self.ent2id[t])
                )

        self.all_true_triples = set(
            self.train_triples + self.valid_triples + self.test_triples
        )

    def get_entity_related_triples(self):
        self.head_related_triples = ddict(list) 
        self.tail_related_triples = ddict(list) 
        self.entity_related_triples = ddict(list) 
        self.entity_related_triples_np = {}
        self.relation_related_triples = ddict(list)
        self.relation_related_triples_np = {}
        self.head_relations_set = ddict(set) 
        self.tail_relations_set = ddict(set) 
        max_len = 0
        
        if self.args.if_pretrain:
            visible_triples = self.train_triples + self.valid_triples + self.test_triples
        else:
            visible_triples = self.train_triples
        for triple in visible_triples: 
            hid,rid,tid = triple
            self.head_related_triples[hid].append(triple)
            self.entity_related_triples[hid].append(triple)
            self.tail_related_triples[tid].append(triple)
            self.entity_related_triples[tid].append(triple)
            self.relation_related_triples[rid].append(triple)
            self.head_relations_set[hid].add(rid)
            self.tail_relations_set[tid].add(rid)
            if max(len(self.head_relations_set[hid]), len(self.tail_relations_set[tid])) > max_len:
                max_len =  max(len(self.head_relations_set[hid]), len(self.tail_relations_set[tid]))
        
        # for key in self.head_related_triples:
        #     self.head_related_triples_np[key] = np.array(self.head_related_triples[key])
        # for key in self.tail_related_triples:
        #     self.tail_related_triples_np[key] = np.array(self.tail_related_triples[key])
        for key in self.entity_related_triples:
            self.entity_related_triples_np[key] = np.array(self.entity_related_triples[key])
        for key in self.relation_related_triples:
            self.relation_related_triples_np[key] = np.array(self.relation_related_triples[key])

 
    def get_2_hop_related_triples(self, specific_entity=None): 
        two_hop_triple_path = os.path.join(self.args.data_path, "2hop_related_triples.pkl")
        if os.path.exists(two_hop_triple_path): 
            print('The two_hop_triple_path exists in', two_hop_triple_path, 'load it.')
            with open(two_hop_triple_path, 'rb') as f:
                self.ent_2_hop_triples = pkl.load(f)        
                            
        else:
            print('The two_hop_triple_path not exists, generate and dump it.')
            self.ent_2_hop_triples = {} 
            # {
            # 0:{(0,r,5):set([(5,r,xx),(xx,r,5)]), (6,r,0):set([(6,r,xx),(xx,r,6)])}, 
            # 1:{()},...}
            if specific_entity:
                searchfor_entity = specific_entity
            else:
                searchfor_entity = self.ent2id.values()
            for ent in searchfor_entity: 
                realted_triple_neighbors=ddict(set) 
                # import pdb; pdb.set_trace()
                for triple in self.head_related_triples[ent]: # (0,r,5)
                    h,r,the_t = triple
                    t_as_head_neighbor = set(self.head_related_triples[the_t]) 
                    t_as_tail_neighbor = set(self.tail_related_triples[the_t]) 
                    realted_triple_neighbors[triple] |= (t_as_head_neighbor| t_as_tail_neighbor) 
                    realted_triple_neighbors[triple].remove(triple)

                for triple in self.tail_related_triples[ent]:# (6,r,0)
                    the_h, r, t = triple
                    h_as_head_neighbor = set(self.head_related_triples[the_h]) 
                    h_as_tail_neighbor = set(self.tail_related_triples[the_h]) 
                    realted_triple_neighbors[triple] |= (h_as_head_neighbor| h_as_tail_neighbor) 
                    realted_triple_neighbors[triple].remove(triple)
                
                self.ent_2_hop_triples[ent] = realted_triple_neighbors
                
                # print('ent',ent, len(realted_triple_neighbors), sum([len(i) for i in realted_triple_neighbors.values()]))
                # for the_triple, neighbors in realted_triple_neighbors.items():
                #     print(the_triple)
                #     print(neighbors)
                

            with open(two_hop_triple_path, "wb") as f:
                pkl.dump(self.ent_2_hop_triples, f, protocol=2)

            # import pdb; pdb.set_trace()
            print('Save two_hop_triples in', two_hop_triple_path)
            
    def sample_subgraph_multihop(self, triple_set, hops, count, ans): 
        
        if hops==0:
            return

        add_triple_set = set()
        for triple in triple_set:
            h,r,t = triple
            h_neighbor_out = set(self.tokenizer.head_related_triples[h][0:count])
            h_neighbor_in = set(self.tokenizer.tail_related_triples[h][0:count])
            t_neighbor_out = set(self.tokenizer.head_related_triples[t][0:count])
            t_neighbor_in = set(self.tokenizer.tail_related_triples[t][0:count])
            
            add_triple_set |= (h_neighbor_out | h_neighbor_in | t_neighbor_out | t_neighbor_in)

        ans |= add_triple_set

        self.sample_subgraph_multihop(add_triple_set, hops-1, count, ans)   

    


    
class KGDataset(Dataset):
    def __init__(self, seq_len, tokenizer:KGTokenizer, triples):
        self.tokenizer = tokenizer
        self.seq_len = seq_len 
        self.triples = triples
    
    def _sample_needed_entity(self, e1, mode, sam_type):
        # import pdb; pdb.set_trace()
        if mode == 'head':
            entity2_relation_dict = self.tokenizer.head_relations_set
        elif mode == 'tail':
            entity2_relation_dict = self.tokenizer.tail_relations_set
            
        e1_relation_set = entity2_relation_dict[e1]
        if sam_type == 'pos':
            cnt = 0
            while True:
                e2 = random.randrange(len(self.tokenizer.token2id),len(self.tokenizer.token2id)+len(self.tokenizer.ent2id))
                cnt += 1
                if cnt < 1000: 
                    if e1 == e2:
                        continue
                e2_relation_set = entity2_relation_dict[e2]
                if len(e1_relation_set & e2_relation_set) > 0:
                    return e2
        elif sam_type == 'neg':
            while True:
                e2 = random.randrange(len(self.tokenizer.token2id),len(self.tokenizer.token2id)+len(self.tokenizer.ent2id))
                e2_relation_set = entity2_relation_dict[e2]
                if len(e1_relation_set & e2_relation_set) == 0:
                    return e2
                
    def process_task_0(self,triple): 
        # task 0
        # two entities are related to the same relation or not
        task_id = 0 
        head1, rid1, tail1 = triple
        if random.random() < 0.5: 
            e1 = head1
            if random.random() < 0.5: 
                e2 = self._sample_needed_entity(e1, 'head', 'pos')
                label = [1]
            else: 
                e2 = self._sample_needed_entity(e1, 'head', 'neg')
                label = [0]
        else: 
            e1 = tail1
            if random.random() < 0.5: 
                e2 = self._sample_needed_entity(e1, 'tail', 'pos')
                label = [1]
            else:
                e2 = self._sample_needed_entity(e1, 'tail', 'neg')
                label = [0]
        
        e1_subgraph = self.sample_one_hop_given_entity(e1, int(self.seq_len/2))
        e2_subgraph = self.sample_one_hop_given_entity(e2, int(self.seq_len/2))
        

        sentence = [self.tokenizer.token2id['[PAD]']] 
        sentence_ft = [self.tokenizer.token2id['[CLS]']]
        mask_ft = [1] 
        
        token_index = [0]
        token_types = [2] # entity 0 relation 1 other 2

        for each_triple in e1_subgraph:
            the_h, the_r, the_t = each_triple
            sentence += [the_h, the_r, the_t, self.tokenizer.token2id['[PAD]']]
            sentence_ft += [self.tokenizer.token2id['[PAD]'], self.tokenizer.token2id['[PAD]'], self.tokenizer.token2id['[PAD]'], self.tokenizer.token2id['[SEP]']]
            
            mask_ft += [0, 0, 0, 1]
            
            token_types += [0, 1, 0, 2]
            token_index.append(len(sentence)-1)
        
        
        sentence += [self.tokenizer.token2id['[PAD]']]
        sentence_ft += [self.tokenizer.token2id['[SEP_G]']]

        mask_ft += [1]
        
        token_types += [2]
        token_index.append(len(sentence)-1)   
        spe_g_index = len(sentence)-1
            
        for each_triple in e2_subgraph:
            the_h, the_r, the_t = each_triple
            sentence += [the_h, the_r, the_t, self.tokenizer.token2id['[PAD]']]
            sentence_ft += [self.tokenizer.token2id['[PAD]'], self.tokenizer.token2id['[PAD]'], self.tokenizer.token2id['[PAD]'], self.tokenizer.token2id['[SEP]']]
            
            mask_ft += [0, 0, 0, 1]
            
            token_types += [0, 1, 0, 2]
            token_index.append(len(sentence)-1)
        
        # [CLS] h1 r1 t1 [SEP] h2 r2 t2 [SEP] [SEP_G] h3 r3 t3 [SEP] h4 r4 t4 [SEP] h5 r5 t5 [SEP] 
        #   0   1  2  3    4   5  6  7    8      9    10 11 12  13   14 15 16  17   18 19 20  21
        extended_visible_matrix = extend_visible_matrix_for_two_entity(e1_subgraph, e2_subgraph, token_index, spe_g_index, if_consider_relation = True, if_interference = False)
        
        label = label*len(sentence)
        
        '''assert len(sentence) == len(sentence_ft)
        assert len(sentence) == len(mask_ft)
        assert len(sentence) == len(token_types)
        assert len(sentence) == len(label)
        assert len(sentence) == len(extended_visible_matrix)'''
        
        return sentence, sentence_ft, mask_ft, task_id, label, extended_visible_matrix, token_types, spe_g_index

    def process_task_1(self,triple):
        # task-1
        # masked reltion
        hid, rid, tid = triple
        task_id = 1
        
        
        try:
            h_subgraph = self.sample_two_hop_given_entity(hid, int(self.seq_len/2))
            t_subgraph = self.sample_two_hop_given_entity(tid, int(self.seq_len/2))
            triple_subgraph_list = list(h_subgraph | t_subgraph)[0:self.seq_len] 
        except: 
            h_subgraph = self.sample_one_hop_given_entity(hid, int(self.seq_len/2))
            t_subgraph = self.sample_one_hop_given_entity(tid, int(self.seq_len/2))
        
            triple_subgraph_list = np.concatenate((h_subgraph, t_subgraph),axis=0)
        
        sentence = [self.tokenizer.token2id['[PAD]']]
        sentence_ft = [self.tokenizer.token2id['[CLS]']]
        mask_ft = [1] 
        
        token_types = [2] 
        label = [-1]
        token_index = [0]
        for each_triple in triple_subgraph_list:
            the_h, the_r, the_t = each_triple
            if random.random() < 0.25: # mask 25% ("25%" can be set to other values)
                sentence += [the_h, self.tokenizer.token2id['[PAD]'], the_t, self.tokenizer.token2id['[PAD]']]
                sentence_ft += [self.tokenizer.token2id['[PAD]'], self.tokenizer.token2id['[MASK]'], self.tokenizer.token2id['[PAD]'], self.tokenizer.token2id['[SEP]']]
                mask_ft += [0, 1, 0, 1] 
                label += [-1, the_r-len(self.tokenizer.ent2id)-len(self.tokenizer.token2id), -1, -1] 
            else:
                sentence += [the_h, the_r, the_t, self.tokenizer.token2id['[PAD]']]
                sentence_ft += [self.tokenizer.token2id['[PAD]'], self.tokenizer.token2id['[PAD]'], self.tokenizer.token2id['[PAD]'], self.tokenizer.token2id['[SEP]']]
                mask_ft += [0, 0, 0, 1]
                label += [-1, -1, -1, -1]
                
            token_types += [0, 1, 0, 2]
            token_index.append(len(sentence)-1)
        
        extended_visible_matrix = extend_visible_matrix_for_one_entity(triple_subgraph_list, token_index, if_consider_relation = True)
        
        
        return sentence, sentence_ft, mask_ft, task_id, label, extended_visible_matrix, token_types

    def process_task_2(self,triple):
        # task-2
        # masked entity
        hid, rid, tid = triple
        task_id = 2
        
        try:
            h_subgraph = self.sample_two_hop_given_entity(hid, int(self.seq_len/2))
            t_subgraph = self.sample_two_hop_given_entity(tid, int(self.seq_len/2))
            triple_subgraph_list = list(h_subgraph | t_subgraph)[0:self.seq_len] 

        except: 
            h_subgraph = self.sample_one_hop_given_entity(hid, int(self.seq_len/2))
            t_subgraph = self.sample_one_hop_given_entity(tid, int(self.seq_len/2))
            triple_subgraph_list = np.concatenate((h_subgraph, t_subgraph),axis=0)

        
        sentence = [self.tokenizer.token2id['[PAD]']]
        sentence_ft = [self.tokenizer.token2id['[CLS]']]
        mask_ft = [1] 
        token_types = [2]
        label = [-1]
        token_index = [0]

        for each_triple in triple_subgraph_list:
            the_h, the_r, the_t = each_triple
            prob = random.random()
            if prob < 0.25: 
                prob /= 0.25
                if prob > 0.5:
                    sentence += [self.tokenizer.token2id['[PAD]'], the_r, the_t, self.tokenizer.token2id['[PAD]']]
                    sentence_ft += [self.tokenizer.token2id['[MASK]'], self.tokenizer.token2id['[PAD]'], self.tokenizer.token2id['[PAD]'], self.tokenizer.token2id['[SEP]']]
                    mask_ft += [1, 0, 0, 1]
                    label += [the_h, -1, -1, -1] 
                else:
                    sentence += [the_h, the_r, self.tokenizer.token2id['[PAD]'], self.tokenizer.token2id['[PAD]']]
                    sentence_ft += [self.tokenizer.token2id['[PAD]'], self.tokenizer.token2id['[PAD]'], self.tokenizer.token2id['[MASK]'], self.tokenizer.token2id['[SEP]']]
                    mask_ft += [0, 0, 1, 1]
                    label += [-1, -1, the_t, -1] 
            else:
                sentence += [the_h, the_r, the_t, self.tokenizer.token2id['[PAD]']]
                sentence_ft += [self.tokenizer.token2id['[PAD]'], self.tokenizer.token2id['[PAD]'], self.tokenizer.token2id['[PAD]'], self.tokenizer.token2id['[SEP]']]
                mask_ft += [0, 0, 0, 1]
                label += [-1, -1, -1, -1] 
            token_types += [0, 1, 0, 2]

            token_index.append(len(sentence)-1)
        

        extended_visible_matrix = extend_visible_matrix_for_one_entity(triple_subgraph_list, token_index, if_consider_relation = True)
        
        
        return sentence, sentence_ft, mask_ft, task_id, label, extended_visible_matrix, token_types
  
    def sample_given_relation(self, rel, count):
        related_triples = self.tokenizer.relation_related_triples_np[rel]
        num = len(related_triples)
        shuffle_index = random.sample(range(0, num), min(num, count)) 
        sample_triples = related_triples[shuffle_index]
        return sample_triples

    def sample_one_hop_given_entity(self, entity, count): 
        
        one_hop_list = self.tokenizer.entity_related_triples_np[entity]
        num = len(one_hop_list)
        shuffle_index = random.sample(range(0, num), min(num, count)) 
        subgraph_one_hop = one_hop_list[shuffle_index]
        
        return subgraph_one_hop
    
    def sample_two_hop_given_entity(self, entity, count):
        subgraph_two_hop=set()
        one_hop_list = self.tokenizer.entity_related_triples[entity]
        num = len(one_hop_list)
        
        shuffle_index = random.sample(range(0, num), min(num, 40, count)) 
        for index in shuffle_index:
            one_hop_trpile = tuple(one_hop_list[index]) 
            subgraph_two_hop.add(one_hop_trpile) 
            if len(subgraph_two_hop) >= count:
                return subgraph_two_hop 

            
            neighbors = list(self.tokenizer.ent_2_hop_triples[entity][one_hop_trpile])
            
            num2 = len(neighbors) 
            shuffle_index2 = random.sample(range(0, num2), min(10, num2, count-len(subgraph_two_hop))) 
            for index2 in shuffle_index2:
                subgraph_two_hop.add(neighbors[index2]) 
                if len(subgraph_two_hop) >= count:
                    return subgraph_two_hop
        
        return subgraph_two_hop 
    
    def sample_two_hop_given_entity_onefirst_fixed(self, entity, count):
        one_hop_list = self.tokenizer.entity_related_triples[entity]
        subgraph_two_hop = set([tuple(i) for i in one_hop_list[:count]])

        if len(subgraph_two_hop) >= count:
            return subgraph_two_hop 
        else:
            for one_hop_trpile in one_hop_list:
                one_hop_trpile = tuple(one_hop_trpile)
                for two_hop_triple in list(self.tokenizer.ent_2_hop_triples[entity][one_hop_trpile])[:]:
                    subgraph_two_hop.add(two_hop_triple)
                    if len(subgraph_two_hop) >= count:
                        return subgraph_two_hop 
        
        return subgraph_two_hop 
    

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, index):
        triple = self.triples[index]
        
        # task 0-2
        sentence_0, sentence_ft_0, mask_ft_0, task_id_0, label_0, extended_visible_matrix_0, token_types_0, spe_g_index_0 = self.process_task_0(triple)
        sentence_1, sentence_ft_1, mask_ft_1, task_id_1, label_1, extended_visible_matrix_1, token_types_1 = self.process_task_1(triple)
        sentence_2, sentence_ft_2, mask_ft_2, task_id_2, label_2, extended_visible_matrix_2, token_types_2 = self.process_task_2(triple)
        
        # assert len(sentence_0) == len(token_types_0)
        # assert len(sentence_1) == len(token_types_1)
        # assert len(sentence_2) == len(token_types_2)

        output = (
            sentence_0, sentence_ft_0, mask_ft_0, task_id_0, label_0, extended_visible_matrix_0, token_types_0, spe_g_index_0, 
            sentence_1, sentence_ft_1, mask_ft_1, task_id_1, label_1, extended_visible_matrix_1, token_types_1,
            sentence_2, sentence_ft_2, mask_ft_2, task_id_2, label_2, extended_visible_matrix_2, token_types_2)

        return output
        
        


def generate_visible_matrix(triple_list, if_consider_relation = True): 
    triple_tensor=torch.tensor(triple_list)
    triple_num = triple_tensor.shape[0]
    v1 = triple_tensor[:,0] 
    v2 = triple_tensor[:,2] 
    
    v1_copy_dim0 = v1.unsqueeze(dim=0).repeat(1, triple_num).reshape(-1) 
    v2_copy_dim0 = v2.unsqueeze(dim=0).repeat(1, triple_num).reshape(-1) 
    
    v1_copy_dim1 = v1.unsqueeze(dim=1).repeat(1, triple_num).reshape(-1) 
    v2_copy_dim1 = v2.unsqueeze(dim=1).repeat(1, triple_num).reshape(-1) 
    
    if if_consider_relation:
        r = triple_tensor[:,1] 
        r_copy_dim0 = r.unsqueeze(dim=0).repeat(1, triple_num).reshape(-1) 
        r_copy_dim1 = r.unsqueeze(dim=1).repeat(1, triple_num).reshape(-1) 

        visible_result=(v1_copy_dim1 == v1_copy_dim0)|(v2_copy_dim1 == v1_copy_dim0)|(v1_copy_dim1 == v2_copy_dim0)|(v2_copy_dim1 == v2_copy_dim0)|(r_copy_dim1 == r_copy_dim0)
    else:
        visible_result=(v1_copy_dim1 == v1_copy_dim0)|(v2_copy_dim1 == v1_copy_dim0)|(v1_copy_dim1 == v2_copy_dim0)|(v2_copy_dim1 == v2_copy_dim0)
    
    visible_matrix = torch.reshape(visible_result.unsqueeze(0),((triple_num,triple_num)))
   
    return visible_matrix, triple_num


def extend_visible_matrix_for_one_entity(triple_subgraph_list, token_index, if_consider_relation = True):
    # return torch.ones((len(triple_subgraph_list)*4+1, len(triple_subgraph_list)*4+1))
    
    visible_matrix, triple_num = generate_visible_matrix(triple_subgraph_list, if_consider_relation)  
    extended_visible_matrix = visible_matrix.repeat_interleave(repeats=4, dim=1).repeat_interleave(repeats=4, dim=0)
    extended_visible_matrix = torch.cat((torch.ones((extended_visible_matrix.shape[0],1)),extended_visible_matrix),dim=1)
    extended_visible_matrix = torch.cat((torch.ones((1,extended_visible_matrix.shape[1])),extended_visible_matrix),dim=0)
    token_index = torch.tensor(token_index)
    extended_visible_matrix[token_index]=1
    extended_visible_matrix[:,token_index]=1

    return extended_visible_matrix

def extend_visible_matrix_for_two_entity(e1_subgraph, e2_subgraph, token_index, spe_g_index, if_consider_relation = True, if_interference = False): 
    # return torch.ones(((len(e1_subgraph)+len(e2_subgraph))*4+2, (len(e1_subgraph)+len(e2_subgraph))*4+2))
    
    if if_interference: 
        triple_subgraph_list = e1_subgraph + e2_subgraph
        visible_matrix, triple_num = generate_visible_matrix(triple_subgraph_list, if_consider_relation)  
        extended_visible_matrix = visible_matrix.repeat_interleave(repeats=4, dim=1).repeat_interleave(repeats=4, dim=0)
        
        extended_visible_matrix = torch.cat((torch.ones((extended_visible_matrix.shape[0],1)),extended_visible_matrix),dim=1)
        extended_visible_matrix = torch.cat((torch.ones((1,extended_visible_matrix.shape[1])),extended_visible_matrix),dim=0)
        
        extended_visible_matrix2 = torch.ones((extended_visible_matrix.shape[0]+1, extended_visible_matrix.shape[1]+1))
        extended_visible_matrix2[0:spe_g_index,0:spe_g_index] =  extended_visible_matrix[0:spe_g_index,0:spe_g_index] 
        extended_visible_matrix2[spe_g_index+1:,0:spe_g_index] =  extended_visible_matrix[spe_g_index:,0:spe_g_index] 
        extended_visible_matrix2[0:spe_g_index,spe_g_index+1:] =  extended_visible_matrix[0:spe_g_index,spe_g_index:] 
        extended_visible_matrix2[spe_g_index+1:,spe_g_index+1:] =  extended_visible_matrix[spe_g_index:,spe_g_index:] 

        token_index = torch.tensor(token_index)
        extended_visible_matrix2[token_index] = 1
        extended_visible_matrix2[:,token_index] = 1

    else: 
        visible_matrix0, triple_num = generate_visible_matrix(e1_subgraph, if_consider_relation)  
        visible_matrix1, triple_num = generate_visible_matrix(e2_subgraph, if_consider_relation)
        extended_visible_matrix0 = visible_matrix0.repeat_interleave(repeats=4, dim=1).repeat_interleave(repeats=4, dim=0)
        extended_visible_matrix1 = visible_matrix1.repeat_interleave(repeats=4, dim=1).repeat_interleave(repeats=4, dim=0)
        
        
        extended_visible_matrix0 = torch.cat((torch.ones((extended_visible_matrix0.shape[0],1)),extended_visible_matrix0),dim=1)
        extended_visible_matrix0 = torch.cat((torch.ones((1,extended_visible_matrix0.shape[1])),extended_visible_matrix0),dim=0)
        
        token_index0 = torch.tensor([i for i in token_index if i < spe_g_index])
        extended_visible_matrix0[token_index0]=1
        extended_visible_matrix0[:,token_index0]=1

        extended_visible_matrix1 = torch.cat((torch.ones((extended_visible_matrix1.shape[0],1)),extended_visible_matrix1),dim=1)
        extended_visible_matrix1 = torch.cat((torch.ones((1,extended_visible_matrix1.shape[1])),extended_visible_matrix1),dim=0)
        
        token_index1 = torch.tensor([i-spe_g_index for i in token_index if i >= spe_g_index])
        extended_visible_matrix1[token_index1]=1
        extended_visible_matrix1[:,token_index1]=1
        
        size0 = extended_visible_matrix0.shape[0]
        size1 = extended_visible_matrix1.shape[0]
        row_col_size =  size0 + size1
        extended_visible_matrix2 = torch.zeros((row_col_size, row_col_size))
        extended_visible_matrix2[0:size0,0:size0] = extended_visible_matrix0
        extended_visible_matrix2[size0:,size0:] = extended_visible_matrix1

        # globel 1: CLS, SEP_G 
        # token_index_global = torch.tensor([0,spe_g_index]) #torch.tensor(token_index)
        # extended_visible_matrix2[token_index_global]=1
        # extended_visible_matrix2[:,token_index_global]=1

    return extended_visible_matrix2


class KGDataset_down_triplecls(KGDataset):
    def __init__(self, seq_len, tokenizer:KGTokenizer, samples, args=None):
        self.tokenizer = tokenizer
        self.seq_len = seq_len 
        self.samples = samples # 
        self.args = args
    
    def generate_sentences_triplecls(self, hid, rid, tid, if_fixed = True): 
        if if_fixed:
            if hid in self.tokenizer.entity_subgraph_fixed:
                h_subgraph = self.tokenizer.entity_subgraph_fixed[hid]
                h_subgraph = h_subgraph.tolist()
            else:
                h_subgraph = self.sample_two_hop_given_entity_onefirst_fixed(hid, int(self.seq_len/2))
                if h_subgraph:
                    h_subgraph = list(h_subgraph)[0:int(self.seq_len/2)]
                else:
                    h_subgraph = [(hid, rid, tid)]
                self.tokenizer.entity_subgraph_fixed[hid] = np.array(h_subgraph)
            if tid in self.tokenizer.entity_subgraph_fixed:
                t_subgraph = self.tokenizer.entity_subgraph_fixed[tid]
                t_subgraph = t_subgraph.tolist()
            else: 
                t_subgraph = self.sample_two_hop_given_entity_onefirst_fixed(tid, int(self.seq_len/2))
                if t_subgraph:
                    t_subgraph = list(t_subgraph)[0:int(self.seq_len/2)]
                else:
                    t_subgraph = [(hid, rid, tid)]
                self.tokenizer.entity_subgraph_fixed[tid] = np.array(t_subgraph)
        else:
            h_subgraph = self.sample_two_hop_given_entity(hid, int(self.seq_len/2))
            if h_subgraph:
                h_subgraph =  list(h_subgraph)[0:int(self.seq_len/2)]
            else:
                h_subgraph = [(hid, rid, tid)]
            t_subgraph = self.sample_two_hop_given_entity(tid, int(self.seq_len/2))
            if t_subgraph:
                t_subgraph = list(t_subgraph)[0:int(self.seq_len/2)]
            else:
                t_subgraph = [(hid, rid, tid)]
            
        
        sentence = [self.tokenizer.token2id['[PAD]']] 
        sentence_ft = [self.tokenizer.token2id['[CLS]']]
        mask_ft = [1] 
        token_types = [2] 

        token_index = [0]
        

        for each_triple in h_subgraph:
            the_h, the_r, the_t = each_triple
            sentence += [the_h, the_r, the_t, self.tokenizer.token2id['[PAD]']]
            sentence_ft += [self.tokenizer.token2id['[PAD]'], self.tokenizer.token2id['[PAD]'], self.tokenizer.token2id['[PAD]'], self.tokenizer.token2id['[SEP]']]
            mask_ft += [0, 0, 0, 1]
            token_types += [0, 1, 0, 2]
            token_index.append(len(sentence)-1)
        
        
        sentence += [self.tokenizer.token2id['[PAD]']]
        sentence_ft += [self.tokenizer.token2id['[SEP_G]']]
        mask_ft += [1]
        token_types += [2]
        token_index.append(len(sentence)-1)
        spe_g_index = len(sentence)-1

        for each_triple in t_subgraph:
            the_h, the_r, the_t = each_triple
            sentence += [the_h, the_r, the_t, self.tokenizer.token2id['[PAD]']]
            sentence_ft += [self.tokenizer.token2id['[PAD]'], self.tokenizer.token2id['[PAD]'], self.tokenizer.token2id['[PAD]'], self.tokenizer.token2id['[SEP]']]
            mask_ft += [0, 0, 0, 1]
            token_types += [0, 1, 0, 2]
            token_index.append(len(sentence)-1)

        try:
            extended_visible_matrix0 = extend_visible_matrix_for_two_entity(h_subgraph, t_subgraph, token_index, spe_g_index, if_consider_relation = True, if_interference = False)
        except:
            import pdb; pdb.set_trace()
        
        sentence += [self.tokenizer.token2id['[PAD]'], hid, rid, tid, self.tokenizer.token2id['[PAD]']]
        sentence_ft += [self.tokenizer.token2id['[TASK]'], self.tokenizer.token2id['[PAD]'], self.tokenizer.token2id['[PAD]'], self.tokenizer.token2id['[PAD]'], self.tokenizer.token2id['[SEP]']]
        mask_ft += [1, 0, 0, 0, 1]
        token_types += [2, 0, 1, 0, 2]
        task_index = len(sentence) - 5 

        size0 = extended_visible_matrix0.shape[0]
        extended_visible_matrix2 = torch.ones((size0 + 5, size0 + 5))
        extended_visible_matrix2[0:size0,0:size0] = extended_visible_matrix0

        assert len(sentence) == len(sentence_ft)
        assert len(sentence) == len(mask_ft)
        assert len(sentence) == len(token_types)
        assert len(sentence) == len(extended_visible_matrix2)

        
        return sentence, sentence_ft, mask_ft, task_index, extended_visible_matrix2, token_types


    def process_task_cls(self, sample):
        if len(sample)==3: 
            sentence_list = []
            sentence_ft_list = []
            mask_ft_list = []
            task_index_list = []
            label_list = []
            extended_visible_matrix_list = []
            token_types_list = []

            hid, rid, tid = sample 
            
            if random.random() < 0.5: 
                sentence, sentence_ft, mask_ft, task_index, extended_visible_matrix, token_type = self.generate_sentences_triplecls(hid, rid, tid, False)
            else:
                sentence, sentence_ft, mask_ft, task_index, extended_visible_matrix, token_type = self.generate_sentences_triplecls(hid, rid, tid)
            
            sentence_list.append(sentence)
            sentence_ft_list.append(sentence_ft)
            mask_ft_list.append(mask_ft)
            task_index_list.append(task_index)
            label_list.append(1)
            extended_visible_matrix_list.append(extended_visible_matrix)
            token_types_list.append(token_type)
            

            neg_tids = random.sample(self.tokenizer.candi_entity_ids, self.args.neg_count)
            neg_hids = random.sample(self.tokenizer.candi_entity_ids, self.args.neg_count)
            for neg_tid in neg_tids:    
                if ((hid, rid, neg_tid) not in self.tokenizer.all_true_triples):
                    if random.random() < 0.5: 
                        sentence_neg, sentence_ft_neg, mask_ft_neg, task_index_neg, extended_visible_matrix_neg, token_type_neg = self.generate_sentences_triplecls(hid, rid, neg_tid, False)
                    else:
                        sentence_neg, sentence_ft_neg, mask_ft_neg, task_index_neg, extended_visible_matrix_neg, token_type_neg = self.generate_sentences_triplecls(hid, rid, neg_tid)
                    sentence_list.append(sentence_neg)
                    sentence_ft_list.append(sentence_ft_neg)
                    mask_ft_list.append(mask_ft_neg)
                    task_index_list.append(task_index_neg)
                    label_list.append(0)
                    extended_visible_matrix_list.append(extended_visible_matrix_neg)
                    token_types_list.append(token_type_neg)

            for neg_hid in neg_hids:
                if ((neg_hid, rid, tid) not in self.tokenizer.all_true_triples):
                    if random.random() < 0.5: 
                        sentence_neg, sentence_ft_neg, mask_ft_neg, task_index_neg, extended_visible_matrix_neg, token_type_neg = self.generate_sentences_triplecls(neg_hid, rid, tid, False)
                    else:
                        sentence_neg, sentence_ft_neg, mask_ft_neg, task_index_neg, extended_visible_matrix_neg, token_type_neg = self.generate_sentences_triplecls(neg_hid, rid, tid)
                    sentence_list.append(sentence_neg)
                    sentence_ft_list.append(sentence_ft_neg)
                    mask_ft_list.append(mask_ft_neg)
                    task_index_list.append(task_index_neg)
                    label_list.append(0)
                    extended_visible_matrix_list.append(extended_visible_matrix_neg)
                    token_types_list.append(token_type_neg)
                
            return sentence_list, sentence_ft_list, mask_ft_list, label_list, extended_visible_matrix_list, task_index_list, token_types_list

        elif len(sample) ==4: 
            hid, rid, tid, label = sample 
            sentence, sentence_ft, mask_ft, task_index, extended_visible_matrix, token_type = self.generate_sentences_triplecls(hid, rid, tid)
            return [sentence], [sentence_ft], [mask_ft], [label], [extended_visible_matrix], [task_index], [token_type]
 
            
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        triple = self.samples[index]

        sentence_list, sentence_ft_list, mask_ft_list, label_list, extended_visible_matrix_list, task_index_list, token_type_list = self.process_task_cls(triple)

        output = (
            sentence_list, sentence_ft_list, mask_ft_list, label_list, extended_visible_matrix_list, task_index_list, token_type_list)

        return output


class KGDataset_down_qa(KGDataset):
    def __init__(self, tokenizer, num_choice, tri_list, qids, labels, *encoder_data):
        self.tokenizer = tokenizer
        self.num_choice = num_choice
        self.tri_list = tri_list
        self.qids = qids
        self.labels = labels
        self.encoder_data = encoder_data

    def __len__(self):
        return len(self.qids)

    def __getitem__(self, index):

        sentence_list,sentence_ft_list, sentence_ent_list, mask_ft_list, token_types_list, mask_index_list, visible_matrix_list, f_index_list = [], [], [], [], [], [], [], []

        curr_tri_list = self.tri_list[index*self.num_choice: index*self.num_choice+self.num_choice]
        for c_subgraph, qamask_list in curr_tri_list:
            
            
            sentence = [self.tokenizer.token2id['[PAD]']] 
            sentence_ft = [self.tokenizer.token2id['[CLS]']] 
            sentence_ent = [self.tokenizer.token2id['[PAD]']] 

            mask_ft = [0] 
            token_types = [2] 
            token_index = [0]  

            for each_triple in c_subgraph:
                the_h, the_r, the_t = each_triple
                sentence += [self.tokenizer.token2id['[PAD]'], the_r, self.tokenizer.token2id['[PAD]'], self.tokenizer.token2id['[PAD]']]
                sentence_ft += [self.tokenizer.token2id['[PAD]'], self.tokenizer.token2id['[PAD]'], self.tokenizer.token2id['[PAD]'], self.tokenizer.token2id['[SEP]']]
                sentence_ent += [the_h, self.tokenizer.token2id['[PAD]'], the_t, self.tokenizer.token2id['[PAD]']]
                mask_ft += [1, 0, 1, 0]
                token_types += [0, 1, 3, 2]
                token_index.append(len(sentence)-1)

            if len(c_subgraph) > 0:
                extended_visible_matrix0 = extend_visible_matrix_for_one_entity(c_subgraph, token_index, if_consider_relation=True)
            else:
                extended_visible_matrix0 = torch.tensor([[1]])

            
            sentence += [self.tokenizer.token2id['[PAD]'], self.tokenizer.token2id['[PAD]'],
                         #self.tokenizer.token2id['[PAD]'], self.tokenizer.token2id['[PAD]'],
                         self.tokenizer.token2id['[PAD]']]
            sentence_ft += [self.tokenizer.token2id['[TASK]'], 
                            # self.tokenizer.token2id['[ENT1]'], self.tokenizer.token2id['[ENT1]'], 
                            self.tokenizer.token2id['[PAD]'],
                         self.tokenizer.token2id['[SEP]']]
            sentence_ent += [self.tokenizer.token2id['[PAD]'], self.tokenizer.token2id['[PAD]'], self.tokenizer.token2id['[PAD]']]
            prompt_len = 3
            mask_index = len(sentence) - prompt_len  # position for [TASK]
            f_index = len(sentence) - 2
            mask_ft += [0]*prompt_len 
            token_types += [2, 4, 2] # 4 qa token [0, 1, 3, 2] head rel tail token qa
                
            # import pdb; pdb.set_trace()
            size0 = extended_visible_matrix0.shape[0]
            extended_visible_matrix2 = torch.zeros((size0 + prompt_len, size0 + prompt_len))  
            extended_visible_matrix2[0:size0, 0:size0] = extended_visible_matrix0

            
            qamask_list = qamask_list.reshape(-1, 1).repeat(4, 1).flatten()  
            qamask_list = np.concatenate([np.array([True]),  
                                         qamask_list, np.array([True]*prompt_len)])  

            
            extended_visible_matrix2[qamask_list, -prompt_len:] = 1.0 
            extended_visible_matrix2[-prompt_len:, qamask_list] = 1.0 
            extended_visible_matrix2[-prompt_len:,-prompt_len:] = 1.0 


            assert len(sentence) == len(sentence_ft)
            assert len(sentence) == len(sentence_ent)
            assert len(sentence) == len(mask_ft)
            assert len(sentence) == len(token_types)
            assert len(sentence) == len(extended_visible_matrix2)


            sentence_list.append(sentence)
            sentence_ft_list.append(sentence_ft)
            sentence_ent_list.append(sentence_ent)
            mask_ft_list.append(mask_ft)
            token_types_list.append(token_types)
            mask_index_list.append(mask_index)
            visible_matrix_list.append(extended_visible_matrix2)
            f_index_list.append(f_index)

        label = self.labels[index]
        encoder_data = [e[index] for e in self.encoder_data]
        
        assert len(sentence_list) == len(sentence_ft_list)
        assert len(sentence_list) == len(sentence_ent_list)
        assert len(sentence_list) == len(mask_ft_list)
        assert len(sentence_list) == len(token_types_list)

        return sentence_list, sentence_ft_list, sentence_ent_list, mask_ft_list, token_types_list, mask_index_list, visible_matrix_list, f_index_list, \
                    label, encoder_data

 
 
class KGDataset_down_zsl_multi_pic(KGDataset):
    def __init__(self, seq_len, tokenizer:KGTokenizer, items, args, if_fixed = False):
        self.tokenizer = tokenizer 
        self.seq_len = seq_len 
        self.items = items
        self.args = args
        self.if_fixed = if_fixed

    def generate_sentences_zsl(self, cid, len_of_piclist, if_fixed): 
        if if_fixed: 
            if cid in self.tokenizer.entity_subgraph_fixed:
                sentence, sentence_ft, mask_ft, token_types, token2fcls, task_index, extended_visible_matrix2, f_index = self.tokenizer.entity_subgraph_fixed[cid]
                sentence = sentence.tolist()
                sentence_ft = sentence_ft.tolist()
                mask_ft = mask_ft.tolist()
                token_types = token_types.tolist()
                token2fcls = token2fcls.tolist()
                need_generate = False
                need_save = False
            else:
                need_generate = True
                need_save = True
        else: 
            need_generate = True
            need_save = False

        if need_generate:
            if if_fixed:
                c_subgraph = list(self.sample_two_hop_given_entity_onefirst_fixed(cid, self.seq_len))[0:self.seq_len]
            else:
                c_subgraph = list(self.sample_two_hop_given_entity(cid, self.seq_len))[0:self.seq_len]
                
            
            sentence = [self.tokenizer.token2id['[PAD]']]
            sentence_ft = [self.tokenizer.token2id['[CLS]']]
            mask_ft = [1] 
            token_types = [2] 
        
            token_index = [0]
            token2fcls = [-1]

            for each_triple in c_subgraph:
                the_h, the_r, the_t = each_triple
                token2fcls += [self.tokenizer.cid_to_fcls[the_h],-1,self.tokenizer.cid_to_fcls[the_t],-1]
                sentence += [the_h, the_r, the_t, self.tokenizer.token2id['[PAD]']]
                sentence_ft += [self.tokenizer.token2id['[PAD]'], self.tokenizer.token2id['[PAD]'], self.tokenizer.token2id['[PAD]'], self.tokenizer.token2id['[SEP]']]
                mask_ft += [0, 0, 0, 1]
                token_types += [0, 1, 0, 2]
        
                token_index.append(len(sentence)-1)
            
            extended_visible_matrix0 = extend_visible_matrix_for_one_entity(c_subgraph, token_index, if_consider_relation = True)

            
            sentence += [self.tokenizer.token2id['[PAD]'], self.tokenizer.token2id['[PAD]'], self.tokenizer.token2id['[PAD]'], self.tokenizer.token2id['[PAD]']] * len_of_piclist
            sentence_ft += [self.tokenizer.token2id['[TASK]'], self.tokenizer.token2id['[PAD]'], self.tokenizer.token2id['[PAD]'], self.tokenizer.token2id['[PAD]']] * len_of_piclist
            mask_ft += [1, 1, 1, 0] * len_of_piclist
            token_types += [2, 2, 2, 2] * len_of_piclist
        
            token2fcls += [-1, -1, -1, -1] * len_of_piclist
            
            sentence_len = len(sentence)
            task_index = [sentence_len-4*i for i in range(len_of_piclist,0,-1)] 
            f_index = [sentence_len-4*i+3 for i in range(len_of_piclist,0,-1)]
            
            
            size0 = extended_visible_matrix0.shape[0]
            assert sentence_len == size0 + 4*len_of_piclist
            extended_visible_matrix2 = torch.zeros((sentence_len, sentence_len)) 
            extended_visible_matrix2[0:size0,0:size0] = extended_visible_matrix0 
            
            for task_id in task_index:
                extended_visible_matrix2[task_id:task_id+4, task_id:task_id+4] = 1.0    
            
            # single 
            extended_visible_matrix2[size0:,:size0] = 1.0
                
            if need_save:
                self.tokenizer.entity_subgraph_fixed[cid] = (np.array(sentence), np.array(sentence_ft), np.array(mask_ft), np.array(token_types), np.array(token2fcls), task_index, extended_visible_matrix2, f_index)
                
            assert len(sentence) == len(sentence_ft)
            assert len(sentence) == len(mask_ft)
            assert len(sentence) == len(token_types)
            assert len(sentence) == len(extended_visible_matrix2)

            

        return sentence, sentence_ft, mask_ft, token_types, token2fcls, task_index, extended_visible_matrix2, f_index


    def process_task(self, item): 
        # train [129, (29404, 41, 1), (29405, 45, 0), (29406, 24, 0)]
        # test [(37314, 46, 201), (37315, 46, 201), (37316, 46, 201)]


        sentence_list, token2fcls_list, task_index_list, extended_visible_matrix_list, f_index_list, fid_list, fcls_list, label_list = [], [], [], [], [], [], [], []
        sentence_ft_list, mask_ft_list, token_types_list = [], [], []
        if isinstance(item[0],int): 
            cid = item[0]

            sentence, sentence_ft, mask_ft, token_types, token2fcls, task_index, extended_visible_matrix, f_index = self.generate_sentences_zsl(cid, self.args.multi_pic, if_fixed=self.if_fixed)

            sentence_list.append(sentence)
            sentence_ft_list.append(sentence_ft)
            mask_ft_list.append(mask_ft)
            token_types_list.append(token_types)

            token2fcls_list.append(token2fcls)
            task_index_list.append(task_index)
            extended_visible_matrix_list.append(extended_visible_matrix)
            f_index_list.append(f_index)
            
            fid = []
            fcls = []
            label = []
            for eachpic in item[1:]:
                the_fid, the_fcls, the_label = eachpic
                fid.append(the_fid)
                fcls.append(the_fcls)
                label.append(the_label)


            fid_list.append(fid)
            fcls_list.append(fcls)
            label_list.append(label)

            
        else:# len(item) == 4: 

            for each_fcls, each_cname in self.tokenizer.awa2_classid2name.items():
                
                each_cid = self.tokenizer.ent2id[each_cname]
                sentence, sentence_ft, mask_ft, token_types, token2fcls, task_index, extended_visible_matrix, f_index = self.generate_sentences_zsl(each_cid, self.args.multi_pic, if_fixed=False)


                sentence_list.append(sentence)
                sentence_ft_list.append(sentence_ft)
                mask_ft_list.append(mask_ft)
                token_types_list.append(token_types)
                token2fcls_list.append(token2fcls)
                task_index_list.append(task_index)
                extended_visible_matrix_list.append(extended_visible_matrix)
                f_index_list.append(f_index)

                # [(37314, 46, 201), (37315, 46, 201), (37316, 46, 201)]

                fid = []
                fcls = []
                label = []
                for eachpic in item[:]:
                    
                    the_fid, the_fcls, the_entityid = eachpic
                    fid.append(the_fid)
                    fcls.append(the_fcls)
                    if each_cid == the_entityid:
                        assert each_fcls == str(the_fcls)
                        label.append(1)
                    else:
                        label.append(0)
                        

                fid_list.append(fid)
                fcls_list.append(fcls)
                label_list.append(label)
        
        return sentence_list, sentence_ft_list, mask_ft_list, token_types_list, token2fcls_list, task_index_list, extended_visible_matrix_list, f_index_list, fid_list, fcls_list, label_list

    
    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index] 
        
        sentence_list, sentence_ft_list, mask_ft_list, token_types_list, token2fcls_list, task_index_list, extended_visible_matrix_list, f_index_list, fid_list, fcls_list, label_list  = self.process_task(item)

        output = (sentence_list, sentence_ft_list, mask_ft_list, token_types_list, token2fcls_list, task_index_list, extended_visible_matrix_list, f_index_list, fid_list, fcls_list, label_list)

        return output
 
        
class Config:
    def __init__(self,tokenizer) -> None:
        self.tokenizer = tokenizer
