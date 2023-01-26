from tqdm import tqdm

codex_train = './CODEXL/train.txt' #(Q504617	P106	Q482980)
codex_valid = './CODEXL/valid.txt'
codex_test = './CODEXL/test.txt'

WN18RR_train = './WN18RR/train.txt' #(00260881	_hypernym	00260622)
WN18RR_valid = './WN18RR/valid.txt'
WN18RR_test = './WN18RR/test.txt'

FB15K237_train = './FB15K237/train.txt' #(/m/01z5tr	/people/person/nationality	/m/09c7w0)
FB15K237_valid = './FB15K237/valid.txt'
FB15K237_test = './FB15K237/test.txt'

out_train = './BIG/train.txt'
out_valid = './BIG/valid.txt'
out_test = './BIG/test.txt'
out_ent = './BIG/entities.dict'
out_rel = './BIG/relations.dict'


with open (out_train,'w') as f_write:
    for train_file in [codex_train, WN18RR_train, FB15K237_train]:
        with open(train_file, 'r') as f_read:
            for row in tqdm(f_read):
                f_write.write(row)


with open (out_valid,'w') as f_write:
    for valid_file in [codex_valid, WN18RR_valid, FB15K237_valid]:
        with open(valid_file, 'r') as f_read:
            for row in tqdm(f_read):
                f_write.write(row)

with open (out_test,'w') as f_write:
    for test_file in [codex_test, WN18RR_test, FB15K237_test]:
        with open(test_file, 'r') as f_read:
            for row in tqdm(f_read):
                f_write.write(row)

ent_dic = {}
rel_dic = {}

        
for triple_file in [out_train, out_valid, out_test]:
    with open(triple_file ,'r') as f_in:
        for row in tqdm(f_in):
            h,r,t = row.strip().split('\t')
            if h in ent_dic:
                pass
            else:
                ent_dic[h] = len(ent_dic)

            if t in ent_dic:
                pass
            else:
                ent_dic[t] = len(ent_dic)
            
            if r in rel_dic:
                pass
            else:
                rel_dic[r] = len(rel_dic)
            
ent_dic = sorted(ent_dic.items(), key = lambda d:d[1], reverse=False)
rel_dic = sorted(rel_dic.items(), key = lambda d:d[1], reverse=False)

with open(out_ent,'w') as f_ent:
    for name, idx in ent_dic:
        f_ent.write(str(idx)+'\t'+name+'\n')
    
with open(out_rel,'w') as f_rel:
    for name, idx in rel_dic:
        f_rel.write(str(idx)+'\t'+name+'\n')
        

