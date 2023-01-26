import imp
from transformers_local import BertModel, BertConfig
import torch.nn as nn
import torch.nn.functional as F
from data import KGTokenizer
import torch 
import scipy.linalg
import scipy.spatial
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")
from qa_utils import TextEncoder, MLP
import numpy as np
from qa_utils import *

class KGBert(nn.Module):
    def __init__(self, tokenizer:KGTokenizer, args) -> None:
        super().__init__()
        
        self.configuration = BertConfig()
        self.configuration.vocab_size = len(tokenizer.token2id) + len(tokenizer.ent2id) + len(tokenizer.rel2id)
        self.configuration.vocab_size_token = len(tokenizer.token2id) # token embedding need ft
        self.configuration.vocab_size_ent = len(tokenizer.token2id) + len(tokenizer.ent2id)
        self.configuration.pad_token_id = tokenizer.token2id['[PAD]'] 
        self.configuration.hidden_size = args.hidden_size
        self.configuration.num_hidden_layers = args.num_hidden_layers
        self.configuration.intermediate_size = self.configuration.hidden_size*4 
        self.configuration.num_attention_heads = args.num_attention_heads
        self.configuration.type_vocab_size = args.token_types # head_entity relation specific tail_entity
        # self.configuration.hidden_dropout_prob = 0.2 # default0.1
        
        self.encoder = BertModel(self.configuration)
        self.tokenizer = tokenizer
        self.cls0 = CLS_based_calssifier(self.configuration.hidden_size*2, 2) 
        self.cls1 = MASK_based_calssifier(self.configuration.hidden_size, len(self.tokenizer.rel2id))
        # self.cls2 = MASK_based_calssifier(self.configuration.hidden_size, len(self.tokenizer.ent2id))
        self.map_mem = nn.Linear(self.configuration.hidden_size, self.configuration.hidden_size)
        
        # NLL(negative log likelihood) loss
        self.nll = nn.NLLLoss(ignore_index=-1) 
        self.crossentrpy = nn.CrossEntropyLoss(ignore_index = -1) 

        # MSE loss
        self.msel = nn.MSELoss(reduction='mean')
        self.msel_noreduction = nn.MSELoss(reduction='none')
        self.bcel = nn.BCELoss(reduction='mean') 
    
    def forward(self, input, task_type, label, spe_g_indexs):
        outputs = self.encoder(**input)
        
        last_hidden_states = outputs.last_hidden_state 
        pooling_states = outputs.pooler_output

        
        indexs = [] 
        for i in range(3): 
            indexs.append(task_type==i)
        
        loss0, acc0 = self.loss_task0(last_hidden_states[indexs[0]],label[indexs[0]], spe_g_indexs)
        loss1, acc1 = self.loss_task1(last_hidden_states[indexs[1]],label[indexs[1]])
        loss2, acc2 = self.loss_task2(last_hidden_states[indexs[2]],label[indexs[2]])

        return loss0, loss1, loss2, acc0, acc1, acc2

    def loss_task0(self, last_hidden, label, spe_g_indexs):
        if len(label) == 0:
            return torch.tensor(0), torch.tensor(0)
        
        cls_emb = last_hidden[:,0]
        
        sep_g_emb = last_hidden[torch.tensor(range(len(spe_g_indexs))),spe_g_indexs]
        output = self.cls0(torch.cat((cls_emb, sep_g_emb), dim = 1))
        # output = self.cls0(cls_emb)
        label = label[:,0]
        loss = self.nll(output, label) 

        # import pdb; pdb.set_trace()
        output2 = output.detach()
        label_pre = output2.argmax(dim=1)
        acc = torch.sum(label_pre == label)/len(label)

        return loss, acc

    def loss_task1(self, last_hidden, label):
        label_useful = label[label!=-1]
        
        if len(label_useful) == 0:
            return torch.tensor(0),torch.tensor(0)
        
        output = self.cls1(last_hidden)#[4, 97, 11] #label 0-40943
         
        loss = self.nll(output.transpose(1, 2), label)
        output2 = output.detach()
        label_pre = output2.argmax(dim=2)[label!=-1] #[64, 506]
        
        acc = torch.sum(label_pre == label_useful)/len(label_useful)
        
        return loss, acc

    def loss_task2(self, last_hidden, label):
        
        the_index = (label!=-1)
        the_label = label[the_index] 

        if len(the_label) == 0:
            
            return torch.tensor(0),torch.tensor(0)
        
        the_hidden = self.map_mem(last_hidden[the_index])
        
        the_embed = self.encoder.embeddings.word_embeddings(the_label)
        
        if False:
            the_hidden = F.normalize(the_hidden, p=2, dim=1) 
            the_embed = F.normalize(the_embed, p=2, dim=1) 
            cos_sim = torch.mm(the_hidden, the_embed.permute(1,0)) 
            cos_sim = (cos_sim + 1)/ 2 
        else:
            
            cos_sim = torch.mm(the_hidden, the_embed.permute(1,0)) 
            cos_sim = F.sigmoid(cos_sim)
        
        label_matrix = (the_label.unsqueeze(1) == the_label.unsqueeze(0)).float()
        
        
        pos_index = (label_matrix == 1)
        neg_index = (label_matrix == 0)
        
        cos_sim_pos = cos_sim[pos_index]
        cos_sim_neg = cos_sim[neg_index]
        
        loss_pos = self.bcel(cos_sim_pos, label_matrix[pos_index]) 
        loss_neg = self.bcel(cos_sim_neg, label_matrix[neg_index]) 
        
        loss = (loss_pos + loss_neg)/2
        
        
        acc_pos = (cos_sim_pos>0.5).sum()/len(cos_sim_pos)
        acc_neg = (cos_sim_neg<0.5).sum()/len(cos_sim_neg)
        
        acc = (acc_pos + acc_neg)/2

        return loss, acc


class CLS_based_calssifier(nn.Module):
    """
    triple task: degree, 3-class classification model : > < = 
    """
    def __init__(self, hidden_size, num_class):
        """
        :param hidden: BERT model output size
        """
        super().__init__()
        self.linear = nn.Linear(hidden_size, num_class)
        self.softmax = nn.LogSoftmax(dim = -1)

    def forward(self, x): # bs * 768
        return self.softmax(self.linear(x))


class MASK_based_calssifier(nn.Module):
    """
    triple task: mask tail entity, total entity size-class classification 
    """
    def __init__(self, hidden, num_class):
        """
        :param hidden: BERT model output size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, num_class)
        self.softmax = nn.LogSoftmax(dim = -1)

    def forward(self, x): 
        return self.softmax(self.linear(x))



class KGBert_down_tirplecls(KGBert):
    def __init__(self, tokenizer:KGTokenizer, args) -> None:
        super().__init__(tokenizer, args)
        
        self.triplecls = CLS_based_calssifier(self.configuration.hidden_size * 2, 2) 

        
    def forward(self, input, label, task_index):
        outputs = self.encoder(**input)

        last_hidden_states = outputs.last_hidden_state 
        
        loss, acc = self.loss_task_triplecls(last_hidden_states,label, task_index)
        return loss, acc


    def loss_task_triplecls(self, last_hidden, label, task_index):
        task_emb = last_hidden[torch.tensor(range(len(task_index))), task_index]
        cls_emb = last_hidden[:, 0]
        output = self.triplecls(torch.cat((cls_emb,task_emb), dim=1)) 
        loss_pos = self.nll(output[label==1], label[label==1]) 
        loss_neg = self.nll(output[label==0], label[label==0])
        loss = (loss_pos + loss_neg) / 2 

        # import pdb; pdb.set_trace()
        label_pre = output.detach().argmax(dim=1)
        acc = metrics.accuracy_score(label.cpu(), label_pre.cpu()) 

        return loss, acc

    def predict(self, input, label, task_index):
        outputs = self.encoder(**input)

        last_hidden = outputs.last_hidden_state 
        task_emb = last_hidden[torch.tensor(range(len(task_index))), task_index]
        cls_emb = last_hidden[:, 0]
        output = self.triplecls(torch.cat((cls_emb,task_emb), dim=1)) 

        loss = self.nll(output, label)

        label_pre = output.argmax(dim=1)

        return loss, label_pre.cpu().numpy(), label.cpu().numpy()



class KGBert_down_qa(nn.Module):
    def __init__(self, tokenizer: KGTokenizer, args):
        super().__init__()
        self.args = args
        self.text_encoder = TextEncoder(tokenizer.model_name)
        self.trans_encoder = KGBert_down_qa_trans(tokenizer, args)
        self.init_embed = self.trans_encoder.encoder.embeddings.word_embeddings.weight.data
        self.init_embed_roberta = self.text_encoder.module.embeddings.word_embeddings.weight.data

    def loss_task_qa(self, logits, labels):
        logits = logits.unsqueeze(1).reshape(-1, self.trans_encoder.tokenizer.num_choice)
        
        loss = self.trans_encoder.loss_func(logits, labels)
        
        return loss

    def qa2feat(self, batch_encoder_data):
        qa_feat_list = []
        for data in batch_encoder_data:
            sent_vecs, all_hidden_states = self.text_encoder(*data, layer_id=-1)
            qa_feat_list.append(sent_vecs)

        qa_feat_list = torch.cat(qa_feat_list)

        return qa_feat_list

    def forward(self, input, label, mask_index, f_index, istrain=True):
        feat = input['features']

        if self.args.train_part=='roberta':
            logits = self.trans_encoder.fc1(self.trans_encoder.dropout_fc(feat))
        else:
            input['features'] = self.trans_encoder.activation(self.trans_encoder.mapping(input['features'])) 
            outputs = self.trans_encoder.encoder(**input)
            # query
            if True:
                idx_range = torch.arange(f_index.shape[0]).to(f_index)
                feat_output = outputs.last_hidden_state[idx_range, f_index, :]
            # cls
            if False:
                cls_output = outputs.last_hidden_state[:, 0, :]

            if True:# pool as qa gnn
                
                the_mask = (1-((input['input_ids_ent']!=7).float() + (input['token_type_ids']==4).float())).bool()
                cls_output, pool_attn = self.trans_encoder.pooler_qagnn(feat, outputs.last_hidden_state, the_mask) 
                
            if False:
                idx_range = torch.arange(f_index.shape[0]).to(f_index)
                cls_output = outputs.last_hidden_state[idx_range, mask_index, :]
            
            
            if False: 
                # import pdb; pdb.set_trace()
                useful_token = (input['input_ids_ft']==1)
                input_mask_expanded = useful_token.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                t = outputs.last_hidden_state * input_mask_expanded
                sum_embeddings = torch.sum(t, 1) # 320 * 768
                sum_mask = useful_token.sum(1) #320
                sum_mask = torch.clamp(sum_mask, min=1e-9)
                cls_output = sum_embeddings/sum_mask.unsqueeze(1)

            
            concat = self.trans_encoder.dropout_fc(torch.cat((cls_output, feat_output, feat), 1))
            # concat = self.trans_encoder.dropout_fc(torch.cat((cls_output, feat), 1))
            # concat = self.trans_encoder.dropout_fc(cls_output)
            logits = self.trans_encoder.fc(concat)
            
        if istrain: 
            loss = self.loss_task_qa(logits, label)
            return loss
        else:
            return logits

    



class KGBert_down_zsl_multi_pic(KGBert):
    def __init__(self, tokenizer:KGTokenizer, args) -> None:
        super().__init__(tokenizer, args)
        self.args = args
        self.mapping = nn.Linear(2048, self.configuration.hidden_size) 
        self.mapping2 = nn.Linear(self.configuration.hidden_size, self.configuration.hidden_size) 
        
        self.mapping2_token = nn.Linear(self.configuration.hidden_size, self.configuration.hidden_size) 
        self.match_bicls = nn.Linear(self.configuration.hidden_size*2, 2)

        self.TokenCLSmapping = nn.Linear(self.configuration.hidden_size*2, tokenizer.attribute_kg.shape[0]) 
        
        self.match = self.TokenCLSmapping

        # self.ent_linear = nn.Linear(self.configuration.hidden_size, tokenizer.attribute_kg.shape[0]) # 50
        self.ent_linear = MLP(self.configuration.hidden_size,self.configuration.hidden_size, tokenizer.attribute_kg.shape[0], 0, 0.2, layer_norm=True)
        self.pic_linear = MLP(self.configuration.hidden_size,self.configuration.hidden_size, tokenizer.attribute_kg.shape[0], 0, 0.2, layer_norm=True)
        
        # self.attribute_layer = nn.Linear(85,50) 

        # self.match2token_1sig = nn.Linear(self.configuration.hidden_size*2, 1) 
        self.init_embed = self.encoder.embeddings.word_embeddings.weight.data
        self.bcel = nn.BCELoss(reduction='none') 



    def forward(self, input, token2fcls, label, task_index, f_index, fcls, unseen_cls= None, seen_cls=None, is_train = True):
        input['features'] = self.mapping(input['features']) # 2048 --> 768

        outputs = self.encoder(**input)
        last_hidden_states = outputs.last_hidden_state 

        loss, acc, loss_pos, loss_neg, pos_acc, neg_acc, all_scores = self.loss_task(last_hidden_states, token2fcls, label, task_index, f_index, input['features'], fcls, is_train)
        if is_train:
            return loss, loss_pos, loss_neg, pos_acc.item(), neg_acc.item(), acc.item()

        else:
            # import pdb; pdb.set_trace()
            all_scores = all_scores.reshape(-1,50,self.args.multi_pic) # 8 * 50 * 3 
            true_label = fcls.reshape(-1,50,self.args.multi_pic)[:,0,:].cpu().numpy() #
            all_scores[:,seen_cls] -= self.args.lmbd # 8 * 40 * 15
            unseen_pre = unseen_cls[all_scores[:,unseen_cls].argmax(dim=1)].cpu().numpy()  
            all_pre = all_scores.argmax(dim=1).cpu().numpy()
            seen_pre = seen_cls[all_scores[:,seen_cls].argmax(dim=1)].cpu().numpy()
            return all_pre.reshape(-1), unseen_pre.reshape(-1), seen_pre.reshape(-1), true_label.reshape(-1)

        
    def loss_task(self, last_hidden, token2fcls, label, task_index, f_index, pic_features, fcls, is_train):
        
        task_index = task_index.reshape(-1)
        label = label.reshape(-1)
        f_index = f_index.reshape(-1)
        fcls = fcls.reshape(-1) 
        
        row_index = torch.tensor(range(last_hidden.shape[0])).unsqueeze(-1).repeat(1,self.args.multi_pic).reshape(-1) 
        

        task_emb = last_hidden[row_index,task_index] # 3*8 768 
        cls_emb = last_hidden[row_index,0]
        f_emb = last_hidden[row_index,f_index]
        
        #f_emb = self.mapping2(f_emb) # 768 --> 768
        #task_emb = self.mapping2_token(task_emb)
        #cls_emb = self.mapping2_token(cls_emb)
        
        
        # import pdb; pdb.set_trace()
        # f_emb = torch.mm(self.pic_linear(f_emb), self.tokenizer.attribute_kg.to(f_emb)) 
        # cls_emb = torch.mm(self.ent_linear(cls_emb), self.tokenizer.attribute_kg.to(cls_emb)) 
        cls_emb = self.ent_linear(cls_emb)
        f_emb = self.pic_linear(f_emb)
        output = torch.mul(cls_emb, f_emb).sum(dim=-1) # 960 = 64 * 15
        
        all_scores = F.sigmoid(output)

        label = label.float()
        pos_idx = label==1.0
        neg_idx = label==0.0
        neg_scores = all_scores[neg_idx]
        pos_scores = all_scores[pos_idx]
        loss_pos = self.bcel(pos_scores, label[pos_idx]).mean() 
        # weight_neg = F.softmax(neg_scores * 2, dim = -1).detach() 
        # loss_neg = (self.bcel(neg_scores, label[neg_idx]) * weight_neg).sum() 
        loss_neg = self.bcel(neg_scores, label[neg_idx]).mean() 

        loss = (loss_pos + loss_neg)/2

        neg_acc = (neg_scores < 0.5).sum()/(neg_idx.sum())
        pos_acc = (pos_scores >= 0.5).sum()/(pos_idx.sum())

        acc = (neg_acc + pos_acc)/2
        return loss, acc, loss_pos, loss_neg, pos_acc, neg_acc, all_scores


