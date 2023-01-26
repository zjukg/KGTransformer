from audioop import avg
from cProfile import label
import imp
import math
from operator import index
from qa_utils import RAdam
from setuptools_scm import Configuration
import torch
import torch.nn as nn
from torch.optim import Adam, Adagrad
from torch.utils.data import DataLoader
import numpy as np
from data import KGTokenizer
from kg_bert import KGBert
from optim_schedule import ScheduledOptim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")
import tqdm
# from transformers_local import get_linear_schedule_with_warmup

try:
    from transformers import (ConstantLRSchedule, WarmupLinearSchedule, WarmupConstantSchedule)
except:
    from transformers import get_constant_schedule, get_constant_schedule_with_warmup,  get_linear_schedule_with_warmup

class KGBertTrainer:
    """
    KGBERTTrainer make the pretrained BERT model with 8 tasks.
    please check the details on README.md with simple example.
    """

    def __init__(self, model, args, logger, tokenizer:KGTokenizer,
                 train_dataloader: DataLoader = None, test_dataloader: DataLoader = None, test_dataloader2: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10):
        """
        :param model: KGBert, KGBert_down_linkpre, ...
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: learning rate of optimizer # 已修改 
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: traning with cuda
        :param log_freq: logging frequency of the batch iteration
        """

        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:"+str(cuda_devices[0]) if cuda_condition else "cpu")

        # Initialize the BERT Language Model, with BERT model
        self.model = model.to(self.device)
        self.args = args
        self.logger = logger
        self.tokenizer = tokenizer
        lr = self.args.lr
        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1 and len(cuda_devices)>1:
            self.logger.info("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader
        self.test_data2 = test_dataloader2

        # Setting the Adam optimizer with hyper-param
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(self.optim, args.hidden_size , n_warmup_steps=warmup_steps)

        self.log_freq = log_freq

        self.logger.info(f"Total Parameters:{sum([p.nelement() for p in self.model.parameters()])}")
        self.logger.info(f"Total transformer Parameters:{sum([p.nelement() for p in self.model.encoder.encoder.parameters()])}")
        

        self.current_metric = 0 
        self.accumulation_steps = 1 
        
    def train(self, epoch):
        self.model.train()
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.model.eval() 
        with torch.no_grad():
            self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch, data_loader, train=True):
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activated
        and also auto save the model every peoch
        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or test
        :return: None
        """
        str_code = "train" if train else "test"

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        avg_loss = 0.0
        avg_loss0 = 0.0
        avg_loss1 = 0.0
        avg_loss2 = 0.0
        avg_acc0 = 0.0
        avg_acc1 = 0.0
        avg_acc2 = 0.0

        task1iter=0 
        task2iter=0

        for i, data in data_iter:
            batch_sentences, batch_sent_ft, batch_mask_ft, mask, token_type_ids, types, labels, final_visible_matrix, spe_g_indexs = data
            # 0. batch_data will be sent into the device(GPU or cpu)
            
            batch_sentences = batch_sentences.to(self.device)
            batch_sent_ft = batch_sent_ft.to(self.device)
            batch_mask_ft = batch_mask_ft.to(self.device)
            mask = mask.to(self.device)
            token_type_ids = token_type_ids.to(self.device)
            labels = labels.to(self.device)
            spe_g_indexs = spe_g_indexs.to(self.device)
            
            final_visible_matrix = final_visible_matrix.to(self.device)

            # 1. forward the subgraph tasks 
            input = {'input_ids': batch_sentences, 'input_ids_ft':batch_sent_ft, 'input_ids_ft_mask':batch_mask_ft, 'token_type_ids': token_type_ids, 'attention_mask': mask, 'final_visible_matrix':final_visible_matrix}
            loss0, loss1, loss2, acc0, acc1, acc2 = self.model(input, types, labels, spe_g_indexs)

            if loss1.item() > 0:
                task1iter +=1
            if loss2.item() > 0:
                task2iter +=1

            # 2. Adding 3 losses 
            loss = self.args.w_t0*loss0 + self.args.w_t1*loss1 + self.args.w_t2*loss2

            # 3. backward and optimization only in train
            if train:
    
                # 2.1 loss regularization
                loss = loss/self.accumulation_steps
                # 2.2 back propagation
                loss.backward()
                  
                
                # 3. update parameters of net
                if((i+1)%self.accumulation_steps)==0:
                # optimizer the net
                    self.optim_schedule.step_and_update_lr()       # update parameters of net
                    self.optim_schedule.zero_grad()   # reset gradien
                    

            avg_loss += loss.item()
            avg_loss0 += loss0.item()
            avg_loss1 += loss1.item()
            avg_loss2 += loss2.item()

            avg_acc0 += acc0.item()
            avg_acc1 += acc1.item()
            avg_acc2 += acc2.item()

        
            post_fix = {
                "epoch": epoch,
                "iter": i,
                "acc0": round(avg_acc0 / (i + 1), 3),
                "acc1": round(avg_acc1 / task1iter, 3),
                "acc2": round(avg_acc2 / task2iter, 3),
                "loss": round(avg_loss * self.accumulation_steps / (i + 1), 3),
                "loss0": round(avg_loss0 / (i + 1), 3),
                "loss1": round(avg_loss1 / task1iter, 3),
                "loss2": round(avg_loss2 / task2iter, 3),
                "iter_loss": round(loss.item() * self.accumulation_steps,3)
            }

            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))

        self.logger.info(f"EP{epoch}_{str_code}, | acc0={(avg_acc0 / len(data_iter)):.3f}, acc1={ (avg_acc1/task1iter):.3f}, acc2={(avg_acc2 / task2iter):.3f} | loss={(avg_loss * self.accumulation_steps / len(data_iter)):.3f}, loss0={(avg_loss0 / len(data_iter)):.3f}, loss1={(avg_loss1 / task1iter):.3f}, loss2={(avg_loss2 / task2iter):.3f}")
        
        if not train: 
            current_metric = avg_loss / len(data_iter)
            self.current_metric = current_metric

    def save(self, epoch, file_path):
        """
        Saving the current BERT model on file_path
        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + ".ep%d" % epoch
        torch.save(self.model.cpu().state_dict(), output_path) 
        self.model.to(self.device)
        self.logger.info(f"EP:{epoch} Model Saved on:{output_path}")
        return output_path




class KGBertTrainer_down_zsl_multi_pic(KGBertTrainer):
    def __init__(self, model, args, logger, tokenizer:KGTokenizer,
                 train_dataloader: DataLoader = None, train_dataloader2:DataLoader = None, test_dataloader: DataLoader = None, test_dataloader2: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10):
        super().__init__(model, args, logger, tokenizer, train_dataloader, test_dataloader, test_dataloader2,
                 lr, betas, weight_decay, warmup_steps, with_cuda, cuda_devices, log_freq)
        
        self.train_dataloader2 = train_dataloader2
        self.lr = self.args.lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.optim = Adam(self.model.parameters(), lr=self.lr, betas=self.betas, weight_decay=self.weight_decay)
        
        #if self.args.continue_pretrain: 
        #if True:
           #self.optim_schedule = ReduceLROnPlateau(self.optim, mode='min', factor=0.5, patience=200, verbose=True, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=1e-6, eps=1e-08)

    def train(self, epoch, model_file_path):
        self.unseen_cls = torch.tensor(self.tokenizer.unseenclasses)
        self.seen_cls = torch.tensor(self.tokenizer.seenclasses)
        self.model.train()
        
        for param_group in self.optim.param_groups:
            # if (epoch+1) % 5 ==0 :
            #     param_group['lr'] *= 0.5
            #     self.train_data = self.train_dataloader2
            current_lr = param_group['lr']
        self.logger.info(f"lr={current_lr}")

        self.iteration_train(epoch, self.train_data)
    
    def test(self, epoch, model_file_path):
        self.unseen_cls = torch.tensor(self.tokenizer.unseenclasses)
        self.seen_cls = torch.tensor(self.tokenizer.seenclasses)
        self.model.eval() 
        with torch.no_grad(): 
            self.iteration_test(epoch, self.test_data, test_type='unseen_set', file_path= model_file_path)

    def test2(self, epoch, model_file_path):
        self.unseen_cls = torch.tensor(self.tokenizer.unseenclasses)
        self.seen_cls = torch.tensor(self.tokenizer.seenclasses)
        self.model.eval()
        with torch.no_grad():
            self.iteration_test(epoch, self.test_data2, test_type='seen_set', file_path= model_file_path)

    def iteration_train(self, epoch, data_loader = None):
        str_code = "train" 

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        avg_loss = 0.0
        avg_loss_p = 0.0
        avg_loss_n = 0.0

        avg_acc = 0.0 
        avg_acc_pos = 0.0
        avg_acc_neg = 0.0


        for i, data in data_iter:
            batch_sentences, batch_sent_ft, batch_mask_ft, batch_token2fcls, mask, token_type_ids, final_visible_matrix, labels, task_indexs, f_indexs, fids, fcls = data

            batch_features = self.tokenizer.features_tensor[fids] # batch*multi_pic*768

            # 0. batch_data will be sent into the device(GPU or cpu)
            batch_sentences = batch_sentences.to(self.device)
            batch_sent_ft = batch_sent_ft.to(self.device)
            batch_mask_ft = batch_mask_ft.to(self.device)
            batch_token2fcls = batch_token2fcls.to(self.device)
            mask = mask.to(self.device)
            token_type_ids = token_type_ids.to(self.device)
            labels = labels.to(self.device)
            task_indexs = task_indexs.to(self.device)
            f_indexs = f_indexs.to(self.device)
            batch_features = batch_features.to(self.device)
            fcls = fcls.to(self.device)
            if final_visible_matrix is not None:
                final_visible_matrix = final_visible_matrix.to(self.device)

            # 1. forward 
            input = {'input_ids': batch_sentences, 
                    'input_ids_ft':batch_sent_ft, 
                    'input_ids_ft_mask':batch_mask_ft,
                    'token_type_ids': token_type_ids, 
                    'attention_mask': mask, 
                    'final_visible_matrix':final_visible_matrix,
                    'feat_indexs':f_indexs, # batch*multi_pic
                    'features':batch_features, # batch*multi_pic*768
                    }
            loss, loss_pos, loss_neg, pos_acc, neg_acc, acc = self.model( input, batch_token2fcls, labels, task_indexs, f_indexs, fcls)

            avg_acc = (avg_acc*i + acc)/(i+1)
            avg_acc_pos = (avg_acc_pos*i + pos_acc)/(i+1)
            avg_acc_neg = (avg_acc_neg*i + neg_acc)/(i+1)
            avg_loss = (avg_loss*i + loss.item())/(i+1)
            avg_loss_p = (avg_loss_p*i + loss_pos.item())/(i+1)
            avg_loss_n = (avg_loss_n*i + loss_neg.item())/(i+1)
            

            # 2. backward and optimization
            #if not self.args.continue_pretrain: # first
            if False:
                self.optim_schedule.zero_grad() 
                loss.backward() 
                self.optim_schedule.step_and_update_lr() 

            else:
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                self.optim = Adam(self.model.parameters(), lr=self.lr, betas=self.betas, weight_decay=self.weight_decay)
                
                # self.optim_schedule.step(loss.item()) 

            post_fix = {
                    "ep": epoch,
                    "iter": i,
                    "loss": round(avg_loss, 3),
                    "loss1_p": round(avg_loss_p, 3),
                    "loss1_n": round(avg_loss_n, 3),
                    'acc':round(avg_acc,3),
                    'acc_pos':round(avg_acc_pos,3),
                    'acc_neg':round(avg_acc_neg,3),
                }
            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))


        self.logger.info(f"EP{epoch}_{str_code}, loss={round(avg_loss,3)}, loss_p={round(avg_loss_p,3)}, loss_n={round(avg_loss_n,3)}, acc={round(avg_acc,3)}, acc_pos={round(avg_acc_pos,3)}, acc_neg={round(avg_acc_neg,3)}")
        

    def iteration_test(self, epoch, data_loader = None, test_type = None, file_path=None):
        
        self.tokenizer.cls_emb = None
        all_pre_list = []
        unseen_pre_list = []
        seen_pre_list = []
        true_label_list = []

        str_code = "test"
        if test_type=='unseen_set':
            useful_count = 7913
        elif test_type=='seen_set':
            useful_count = 5882


        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        pre_output_path = file_path + "_ep%d" % epoch + "_" + test_type + '-mat'
        with open(pre_output_path, 'w') as f_out:
            f_out.write(','.join(['all_pre', 'unseen_pre', 'seen_pre', 'true_label'])+'\n')
            for i, data in data_iter:
                batch_sentences, batch_sent_ft, batch_mask_ft, batch_token2fcls, mask, token_type_ids, final_visible_matrix, labels, task_indexs, f_indexs, fids, fcls = data
                unseen_cls =  self.unseen_cls.to(self.device)
                seen_cls =  self.seen_cls.to(self.device)
                batch_features = self.tokenizer.features_tensor[fids]
                batch_features = batch_features.to(self.device)
                labels = labels.to(self.device)
                fcls = fcls.to(self.device)

                if True: 
                    mask = mask.to(self.device)
                    token_type_ids = token_type_ids.to(self.device)
                    task_indexs = task_indexs.to(self.device)
                    f_indexs = f_indexs.to(self.device)
                    batch_sentences = batch_sentences.to(self.device)
                    batch_sent_ft = batch_sent_ft.to(self.device)
                    batch_mask_ft = batch_mask_ft.to(self.device)
                    batch_token2fcls = batch_token2fcls.to(self.device)
                
                    if final_visible_matrix is not None:
                        final_visible_matrix = final_visible_matrix.to(self.device)

                    # 1. forward the 8 pre_train tasks
                    input = {'input_ids': batch_sentences, 
                            'input_ids_ft':batch_sent_ft, 
                            'input_ids_ft_mask':batch_mask_ft,
                            'token_type_ids': token_type_ids, 
                            'attention_mask': mask, 
                            'final_visible_matrix':final_visible_matrix,
                            'feat_indexs':f_indexs,
                            'features':batch_features,
                            }
                        
                    all_pre, unseen_pre, seen_pre, true_label = self.model(input, batch_token2fcls, labels, task_indexs, f_indexs, fcls, unseen_cls, seen_cls, is_train = False)
                    
            
                for pre1, pre2, pre3, tlabel in zip(all_pre, unseen_pre, seen_pre, true_label):
                    
                    f_out.write(','.join([str(pre1), str(pre2), str(pre3), str(tlabel)])+'\n')
                    all_pre_list.append(pre1)
                    unseen_pre_list.append(pre2)
                    seen_pre_list.append(pre3)
                    true_label_list.append(tlabel)

                    if len(all_pre_list)>=useful_count:
                        break
                
                f_out.flush()  

                
                acc_all = round(metrics.accuracy_score(true_label_list, all_pre_list) ,6)
                acc_unseen = round(metrics.accuracy_score(true_label_list, unseen_pre_list) ,6)
                acc_seen = round(metrics.accuracy_score(true_label_list, seen_pre_list) ,6)


                post_fix = {
                    "ep": epoch,
                    "iter": i,
                    'mode': 'mat',
                    'acc_all':round(acc_all, 3),
                    'acc_unseen':round(acc_unseen, 3),
                    'acc_seen':round(acc_seen, 3),
                }


                if i % self.log_freq == 0:
                    data_iter.write(str(post_fix))
                    
        self.logger.info(f"EP{epoch}_{str_code}, acc_all={round(acc_all,3)}, acc_unseen={round(acc_unseen ,3)}, acc_seen={round(acc_seen ,3)}")
        
        self.current_metric = (acc_all, acc_unseen, acc_seen)


    def save(self, epoch, file_path, metric, value):
        """
        Saving the current BERT model on file_path
        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + f".ep{epoch}_{metric}-{value}"
        torch.save(self.model.cpu().state_dict(), output_path)
        self.model.to(self.device)
        self.logger.info(f"EP:{epoch} Model Saved on:{output_path}")
        return output_path

class KGBertTrainer_down_qa:
    def __init__(self, model, args, logger, tokenizer:KGTokenizer,
                 train_dataloader=None, test_dataloader=None, test_dataloader2=None,
                 with_cuda: bool = True, device0=None, device1=None, log_freq: int = 10):

        self.model = model
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device0 = torch.device("cuda:"+str(device0[0]) if cuda_condition else "cpu")
        self.device1 = torch.device("cuda:"+str(device1[0]) if cuda_condition else "cpu")

        # Initialize the BERT Language Model, with BERT model
        self.args = args
        self.logger = logger
        self.tokenizer = tokenizer

        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader
        self.test_data2 = test_dataloader2


        self.log_freq = log_freq

        self.logger.info(f"Total Parameters:{sum([p.nelement() for p in self.model.parameters()])}")

        self.current_metric = 0 

        
        self.model.trans_encoder = self.model.trans_encoder.to(self.device0)
        self.model.text_encoder = self.model.text_encoder.to(self.device1)
        self.accumulation_steps = self.args.big_bs//self.args.train_bs # 64/16 = 4
        
        # Setting the Adam optimizer with hyper-param
        # import pdb; pdb.set_trace()
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        grouped_parameters = [
            {'params': [p for n, p in self.model.text_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2, 'lr': self.args.encoder_lr},
            {'params': [p for n, p in self.model.text_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': self.args.encoder_lr},
            {'params': [p for n, p in self.model.trans_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2, 'lr': self.args.decoder_lr},
            {'params': [p for n, p in self.model.trans_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': self.args.decoder_lr},
        ]
        self.optim = RAdam(grouped_parameters)
        
        try:       
            self.scheduler = ConstantLRSchedule(self.optim)
        except:
            self.scheduler = get_constant_schedule(self.optim)
        

    def train(self, epoch):
        self.model.train()

        str_code = "train"

        # Setting the tqdm progress bar
        # if epoch%3 == 0:
        #     self.now_data = self.train_data
        # elif epoch%3 == 1:
        #     self.now_data = self.test_data2
        # else:
        #     self.now_data = self.test_data
        self.now_data = self.train_data
        
        data_iter = tqdm.tqdm(enumerate(self.now_data),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(self.now_data),
                              bar_format="{l_bar}{r_bar}")

        avg_loss = 0.0

        for i, data in data_iter:
            batch_loss = 0

            batch_sentences, batch_sent_ft, batch_sent_ent, batch_mask_ft, attention_mask, token_type_ids, final_visible_matrix, \
                batch_mask_index_list, batch_f_index_list, \
                batch_label, batch_encoder_data = data

            # 0. batch_data will be sent into the device(GPU or cpu)
            batch_sentences = batch_sentences.to(self.device0)
            batch_sent_ft = batch_sent_ft.to(self.device0)
            batch_sent_ent = batch_sent_ent.to(self.device0)
            batch_mask_ft = batch_mask_ft.to(self.device0)
            attention_mask = attention_mask.to(self.device0)
            token_type_ids = token_type_ids.to(self.device0)
            final_visible_matrix = final_visible_matrix.to(self.device0)
            batch_mask_index_list = batch_mask_index_list.to(self.device0)
            batch_f_index_list = batch_f_index_list.to(self.device0)

            batch_label = batch_label.to(self.device0)
            batch_encoder_data = [[d.to(self.device1) for d in b] for b in batch_encoder_data]
            # print(batch_sentences.shape)
            # for mi, mdata in enumerate(range(math.ceil(batch_label.size(0) / self.args.train_mini_bs))):
            if True:
                # a = mi*self.args.train_mini_bs
                # b = min((mi+1)*self.args.train_mini_bs, batch_label.size(0))

                # a_ = a*self.tokenizer.num_choice
                # b_ = min(b*self.tokenizer.num_choice, batch_sentences.size(0))

                # batch_qa_features = self.model.qa2feat(batch_encoder_data[a:b])
                batch_qa_features = self.model.qa2feat(batch_encoder_data)
                # 1. forward
            #     input = {
            #         'input_ids': batch_sentences[a_:b_],
            #         'input_ids_ft':batch_sent_ft[a_:b_], 
            #         'input_ids_ft_mask':batch_mask_ft[a_:b_],
            #         'token_type_ids': token_type_ids[a_:b_],
            #         'attention_mask': attention_mask[a_:b_],
            #         'final_visible_matrix': final_visible_matrix[a_:b_],
            #         'feat_indexs': batch_f_index_list[a_:b_],
            #         'features': batch_qa_features.to(self.device0),
            #     }

            #     loss = self.model(input, batch_label[a:b],
            #                       batch_mask_index_list[a_:b_], batch_f_index_list[a_:b_])

            #     batch_loss += (loss*(b-a))

            # batch_loss /= batch_label.size(0)
                input = {
                    'input_ids': batch_sentences,
                    'input_ids_ft':batch_sent_ft, 
                    'input_ids_ent':batch_sent_ent,
                    'input_ids_ft_mask':batch_mask_ft,
                    'token_type_ids': token_type_ids,
                    'attention_mask': attention_mask,
                    'final_visible_matrix': final_visible_matrix,
                    'feat_indexs': batch_f_index_list,
                    'features': batch_qa_features.to(self.device0),
                }
                batch_loss = self.model(input, batch_label, batch_mask_index_list, batch_f_index_list)

            avg_loss = (avg_loss * i + batch_loss.item()) / (i + 1)

            # 2. optimization
            if True:
                # self.optim_schedule.zero_grad() 
                # batch_loss.backward() 
                # self.optim_schedule.step_and_update_lr() 
                
                # 2.1 loss regularization
                batch_loss = batch_loss/self.accumulation_steps
                # 2.2 back propagation
                batch_loss.backward()
  
                # 3. update parameters of net,  
                if((i+1)%self.accumulation_steps)==0:
                    #self.optim_schedule.step_and_update_lr()       # update parameters of net
                    #self.optim_schedule.zero_grad()   # reset gradient
                    self.scheduler.step()
                    self.optim.step()
                    self.optim.zero_grad()
                    # self.optim_schedule.step(avg_loss)
                    # self.optim = Adam(self.model.parameters(), lr=self.args.lr)#, betas=betas, weight_decay=weight_decay)
                    # self.optim = Adam(self.model.parameters(), lr=self.args.lr, betas=self.betas, weight_decay=self.weight_decay)

            else:
                batch_loss.backward()
                self.optim.step()
                self.optim_schedule.step(batch_loss.item())
                self.optim.zero_grad()

            post_fix = {
                "ep": epoch,
                "iter": i,
                "loss": round(avg_loss, 3),
                "batch_loss": round(batch_loss.item(), 3),
            }

            if (i+1) % self.log_freq == 0:
                data_iter.write(str(post_fix))

        self.logger.info(
            f"EP{epoch}_{str_code}, loss={round(avg_loss, 3)}")

    def eval(self, epoch, istest=False):
        self.model.eval()
        if istest:
            str_code = "test"
            data_iter = tqdm.tqdm(enumerate(self.test_data),
                                  desc="EP_%s:%d" % (str_code, epoch),
                                  total=len(self.test_data),
                                  bar_format="{l_bar}{r_bar}")
        else:
            str_code = "valid"
            data_iter = tqdm.tqdm(enumerate(self.test_data2),
                                  desc="EP_%s:%d" % (str_code, epoch),
                                  total=len(self.test_data2),
                                  bar_format="{l_bar}{r_bar}")
        with torch.no_grad():
            n_samples, n_correct = 0, 0
            for i, data in data_iter:
                batch_sentences, batch_sent_ft, batch_sent_ent, batch_mask_ft, attention_mask, token_type_ids, final_visible_matrix, \
                    batch_mask_index_list, batch_f_index_list, \
                    batch_label, batch_encoder_data = data

                # 0. batch_data will be sent into the device(GPU or cpu)
                batch_sentences = batch_sentences.to(self.device0)
                batch_sent_ft = batch_sent_ft.to(self.device0)
                batch_sent_ent = batch_sent_ent.to(self.device0)
                batch_mask_ft = batch_mask_ft.to(self.device0)
                attention_mask = attention_mask.to(self.device0)
                token_type_ids = token_type_ids.to(self.device0)
                final_visible_matrix = final_visible_matrix.to(self.device0)
                batch_mask_index_list = batch_mask_index_list.to(self.device0)
                batch_f_index_list = batch_f_index_list.to(self.device0)

                batch_label = batch_label.to(self.device0)
                batch_encoder_data = [[d.to(self.device1) for d in b] for b in batch_encoder_data]
                batch_qa_features = self.model.qa2feat(batch_encoder_data)

                # 1. forward
                input = {
                    'input_ids': batch_sentences,
                    'input_ids_ft':batch_sent_ft, 
                    'input_ids_ent':batch_sent_ent,
                    'input_ids_ft_mask':batch_mask_ft,
                    'token_type_ids': token_type_ids,
                    'attention_mask': attention_mask,
                    'final_visible_matrix': final_visible_matrix,
                    'feat_indexs': batch_f_index_list,
                    'features': batch_qa_features.to(self.device0)
                }

                logits = self.model(input, batch_label,
                                  batch_mask_index_list, batch_f_index_list,
                                  istrain=False)

                logits = logits.unsqueeze(1).reshape(-1, self.tokenizer.num_choice)

                n_correct += (logits.argmax(1) == batch_label).sum().item()
                n_samples += batch_label.size(0)

                acc = n_correct / n_samples

                post_fix = {
                    "ep": epoch,
                    "iter": i,
                    "acc": round(acc, 3),
                }

                if i % (self.log_freq//5) == 0:
                    data_iter.write(str(post_fix))

            acc = n_correct / n_samples

            self.logger.info(f"EP{epoch}_{str_code}, acc={round(acc, 3)}")

        return acc
        # self.current_metric = acc

    def save(self, epoch, file_path, metric, value):
        """
        Saving the current BERT model on file_path
        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + f".ep{epoch}_{metric}-{value}"
        torch.save(self.model.cpu().state_dict(), output_path)
        self.model.trans_encoder = self.model.trans_encoder.to(self.device0)
        self.model.text_encoder = self.model.text_encoder.to(self.device1)
        self.logger.info(f"EP:{epoch} Model Saved on:{output_path}")
        return output_path


class KGBertTrainer_down_triplecls(KGBertTrainer):
    def __init__(self, model, args, logger, tokenizer:KGTokenizer,
                 train_dataloader: DataLoader = None, test_dataloader: DataLoader = None, test_dataloader2: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10, test_type = None):
        super().__init__(model, args, logger, tokenizer, train_dataloader, test_dataloader, test_dataloader2,
                 lr, betas, weight_decay, warmup_steps, with_cuda, cuda_devices, log_freq)
        
        self.test_type = test_type
        #if self.args.continue_pretrain: 
        if True:
            self.optim_schedule = ReduceLROnPlateau(self.optim, mode='min', factor=0.8, patience=400, verbose=True, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=1e-6, eps=1e-08)

    def train(self, epoch):
        self.model.train()
        self.iteration_train(epoch, self.train_data)
    
    def test(self, epoch, model_file_path):
        self.model.eval() 
        with torch.no_grad():
            self.iteration_test(epoch, self.test_data, file_path= model_file_path)

    def iteration_train(self, epoch, data_loader = None):
        str_code = "train" 

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")


        for i, data in data_iter:
            batch_sentences, batch_sent_ft, batch_mask_ft, mask, token_type_ids, labels, final_visible_matrix, task_indexs = data

            # 0. batch_data will be sent into the device(GPU or cpu)
            batch_sentences = batch_sentences.to(self.device)
            batch_sent_ft = batch_sent_ft.to(self.device)
            batch_mask_ft = batch_mask_ft.to(self.device)
            mask = mask.to(self.device)
            token_type_ids = token_type_ids.to(self.device)
            labels = labels.to(self.device)
            task_indexs = task_indexs.to(self.device)
            
            if final_visible_matrix is not None:
                final_visible_matrix = final_visible_matrix.to(self.device)

            # 1. forward 
            input = {'input_ids': batch_sentences, 
                    'input_ids_ft':batch_sent_ft, 
                    'input_ids_ft_mask':batch_mask_ft,
                    'token_type_ids': token_type_ids, 
                    'attention_mask': mask, 
                    'final_visible_matrix':final_visible_matrix,
                    }

            loss, acc = self.model(input, labels, task_indexs) 

            
            # 2. backward and optimization
            #if not self.args.continue_pretrain: 
            if False:
                self.optim_schedule.zero_grad() 
                loss.backward() 
                self.optim_schedule.step_and_update_lr() 

            else:
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                self.optim_schedule.step(loss.item()) 

            post_fix = {
                    "ep": epoch,
                    "iter": i,
                    "loss": round(loss.item(), 3),
                    'acc':round(acc.item(),3),
                }
            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))

        self.logger.info(f"EP{epoch}_{str_code}, loss={round(loss.item(),3)}, acc={round(acc.item(),3)}")
        

    def iteration_test(self, epoch, data_loader = None, file_path=None):
        #self.tokenizer.cls_emb = None
        all_pre_list = []
        true_label_list = []

        str_code = "test"

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        
        pre_output_path = file_path + "_ep%d" % epoch + "_" + self.test_type 
        with open(pre_output_path, 'w') as f_out:
            
            f_out.write(','.join(['all_pre', 'true_label'])+'\n')
                
            for i, data in data_iter:
                batch_sentences, batch_sent_ft, batch_mask_ft, mask, token_type_ids, labels, final_visible_matrix, task_indexs = data

                # 0. batch_data will be sent into the device(GPU or cpu)
                batch_sentences = batch_sentences.to(self.device)
                batch_sent_ft = batch_sent_ft.to(self.device)
                batch_mask_ft = batch_mask_ft.to(self.device)
                mask = mask.to(self.device)
                token_type_ids = token_type_ids.to(self.device)
                labels = labels.to(self.device)
                task_indexs = task_indexs.to(self.device)
                
                if final_visible_matrix is not None:
                    final_visible_matrix = final_visible_matrix.to(self.device)

                # 1. forward 
                input = {'input_ids': batch_sentences, 
                        'input_ids_ft':batch_sent_ft, 
                        'input_ids_ft_mask':batch_mask_ft,
                        'token_type_ids': token_type_ids, 
                        'attention_mask': mask, 
                        'final_visible_matrix':final_visible_matrix,
                        }
                # print(types)
                loss, all_pre, true_label = self.model.predict(input, labels, task_indexs) 
                
                for pre1, tlabel, in zip(all_pre, true_label):
                    f_out.write(','.join([str(pre1), str(tlabel)])+'\n')
                    all_pre_list.append(pre1)
                    true_label_list.append(tlabel)
                    f_out.flush()  
                    
                acc = round(metrics.accuracy_score(true_label_list, all_pre_list) ,6)
                precision = round(metrics.precision_score(true_label_list, all_pre_list) ,6)
                recall = round(metrics.recall_score(true_label_list, all_pre_list) ,6)
                f1 = round(metrics.f1_score(true_label_list, all_pre_list) ,6)
        
        
                post_fix = {
                    "ep": epoch,
                    "iter": i,
                    'loss': round(loss.item(), 3),
                    'acc':round(acc, 6),
                    'precision':round(precision, 6),
                    'recall':round(recall, 6),
                    'f1':round(f1, 6),
                }

                if i % self.log_freq == 0:
                    data_iter.write(str(post_fix))
                    
                        
        self.logger.info(f"EP{epoch}_{str_code}, loss={round(loss.item(),3)}, acc={round(acc,6)}, precision={round(precision,6)}, recall={round(recall, 6)}, f1={round(f1,6)}")
        
        self.current_metric = f1


    def save(self, epoch, file_path, metric, value):
        """
        Saving the current BERT model on file_path
        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + f".ep{epoch}_{metric}-{value}"
        torch.save(self.model.cpu().state_dict(), output_path)
        self.model.to(self.device)
        self.logger.info(f"EP:{epoch} Model Saved on:{output_path}")
        return output_path
