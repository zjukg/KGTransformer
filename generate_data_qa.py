from tqdm import tqdm
import pickle as pkl
import numpy as np
import torch

train_tri_list_path = 'qa_data/train_tri_list.prune.pkl'
dev_tri_list_path = 'qa_data/dev_tri_list.prune.pkl'
test_tri_list_path = 'qa_data/test_tri_list.prune.pkl'
ent_emb_all_path = 'qa_data/tzw.ent.npy'

train_tri_list = pkl.load(open(train_tri_list_path, 'rb'))
dev_tri_list = pkl.load(open(dev_tri_list_path, 'rb'))
test_tri_list = pkl.load(open(test_tri_list_path, 'rb'))

out_ent = 'dataset/down_qa/entities.dict'
out_ent_emb = 'dataset/down_qa/ent_emb.pt'
out_rel = 'dataset/down_qa/relations.dict'


ent_dic = {}
rel_dic = {}

for triples_list in [train_tri_list, dev_tri_list, test_tri_list]:
    for row in tqdm(triples_list):
        for triple in row[0]:
            h, r, t = triple
            if h not in ent_dic:
                ent_dic[h] = len(ent_dic)

            if t not in ent_dic:
                ent_dic[t] = len(ent_dic)
            
            if r not in rel_dic:
                rel_dic[r] = len(rel_dic)
            
ent_dic = sorted(ent_dic.items(), key = lambda d:d[1], reverse=False)
rel_dic = sorted(rel_dic.items(), key = lambda d:d[1], reverse=False)

ent_emb = np.zeros((len(ent_dic),1024))
ent_emb_all = np.load(ent_emb_all_path)

with open(out_ent,'w') as f_ent:
    for name, idx in ent_dic:
        f_ent.write(str(idx)+'\t'+str(name)+'\n')
        ent_emb[idx] = ent_emb_all[name]

ent_emb = torch.from_numpy(ent_emb)
torch.save(ent_emb, out_ent_emb)

with open(out_rel,'w') as f_rel:
    for name, idx in rel_dic:
        f_rel.write(str(idx)+'\t'+str(name)+'\n')
        
