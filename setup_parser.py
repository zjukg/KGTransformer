# -*- coding: utf-8 -*-
import argparse
from ast import parse
from distutils.log import error
import os
from turtle import down
from wsgiref.simple_server import demo_app
def setup_parser():
    """Set up Python's ArgumentParser with data, model, trainer, and other arguments."""
    parser = argparse.ArgumentParser(add_help=False)

    # Basic arguments
    parser.add_argument('--dataset_name', default="", type=str)
    parser.add_argument('--pretrain_dataset', default="", type=str)
    parser.add_argument('--down_task', default="", type=str)
    
    parser.add_argument('--cuda_devices', default=[0], nargs='+', type=int)
    

    parser.add_argument('--epochs', default=100, type=int)

    parser.add_argument('--train_bs', default=32, type=int)
    # parser.add_argument('--train_mini_bs', default=4, type=int)# qa
    parser.add_argument('--test_bs', default=32, type=int)
    parser.add_argument('--lr', default=1e-4, type = float)

    # 预训练任务权重
    parser.add_argument('--w_t0', default=1.0, type=float)
    parser.add_argument('--w_t1', default=1.0, type=float)
    parser.add_argument('--w_t2', default=1.0, type=float)

    parser.add_argument('--hidden_size', default=768, type=int) 
    parser.add_argument('--num_hidden_layers', default= 4, type=int) 
    parser.add_argument('--num_attention_heads', default= 12, type=int) 
    parser.add_argument('--seq_len_pre', default=126, type=int) 
    parser.add_argument('--seq_len_down', default=126, type=int) 
    parser.add_argument('--seq_down_type', default=2, type=int)  
    parser.add_argument('--textEmb', default='cls', type=str) 

    parser.add_argument('--spl_unseen', default = 0, type=int) 
    parser.add_argument('--fixed_unseen', default = 1, type=int) 
    parser.add_argument('--fixed_seen', default = 1, type=int) 
    parser.add_argument('--loss_type', default = 3, type=int)  
    parser.add_argument('--multi_pic', default = 10, type = int) # number of pic

    
    parser.add_argument('--neg_count', default = 3, type=int) 
    parser.add_argument('--fixedT', default = 0, type=int) 
    parser.add_argument('--lmbd', default=0.3, type=float) 

    parser.add_argument('--continue_pretrain', action='store_true', default=False) 
    parser.add_argument('--direct_ft', action='store_true', default=False) 

    parser.add_argument('--debug', action='store_true', default=False) 
    parser.add_argument('--test_epoch', default=1, type=int) 
    
    parser.add_argument('--log_freq', default=10, type=int)# log

    # for qa
    parser.add_argument('--train_split', default=1, type=int)# 1 epoch only 1/n sample, prevent overfit
    parser.add_argument('--train_part', default='together', type=str) # (roberta, kgtrans, together)
    parser.add_argument('--encoder_lr', default=1e-5, type=float) # roberta 
    parser.add_argument('--decoder_lr', default=1e-4, type=float) # transf
    parser.add_argument('--cuda_devices1', default=[1], nargs='+', type=int) #qa的
    parser.add_argument('--big_bs', default=64, type=int)
    parser.add_argument('--token_types',default=4, type = int) # 0head 1rel 3tail 2token 
    # 4 for qa 
    

    args = parser.parse_args()
    


    args.petrain_save_dir = os.path.join("pretrain_models", args.pretrain_dataset)
    if not os.path.exists(args.petrain_save_dir):
        os.makedirs(args.petrain_save_dir)
    
    args.premodel_name = f'model_layer-{args.num_hidden_layers}_hidden-{args.hidden_size}_heads-{args.num_attention_heads}_seq-{args.seq_len_pre}_textE-{args.textEmb}_t0-{args.w_t0}_t1-{args.w_t1}_t2-{args.w_t2}'
    
    #if args.down_task in ['down_triplecls','down_zsl']:
     #   args.premodel_name += '_'+args.down_task
    
    args.petrain_save_path = os.path.join(args.petrain_save_dir, args.premodel_name)
    args.log_file  = args.petrain_save_path + '.log' 

    # downstream task data
    args.data_path = os.path.join('dataset', args.dataset_name)
    
    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    # downstream task model
    if args.down_task=='down_zsl':
        args.seq_len_down -= args.multi_pic 
        down_model_add_name = f'_fixedT-{args.fixedT}'
    elif args.down_task=='down_triplecls':    
        down_model_add_name = f'_fixedT-{args.fixedT}'
    elif args.down_task =='down_qa':
        down_model_add_name = f'_fixedT-{args.fixedT}_tp-{args.train_part}'
    else:
        down_model_add_name = ''
    
    args.down_task_model_dir = os.path.join('downstream_models', args.down_task, args.dataset_name, args.pretrain_dataset) 
    if not os.path.exists(args.down_task_model_dir):
        os.makedirs(args.down_task_model_dir)
    if args.direct_ft: 
        args.down_task_model_path = os.path.join(args.down_task_model_dir, 
                        f'model_layer-{args.num_hidden_layers}_hidden-{args.hidden_size}_heads-{args.num_attention_heads}_seq-{args.seq_len_pre}_textE-{args.textEmb}_direct_ft' + down_model_add_name)
    else:
        args.down_task_model_path = os.path.join(args.down_task_model_dir, args.premodel_name + down_model_add_name)
    args.log_file_down_task = args.down_task_model_path + '.log'


    return args
