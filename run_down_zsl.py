from tkinter.messagebox import NO
from torch.utils.data import DataLoader
from data import KGTokenizer, Config, KGDataset_down_zsl_multi_pic
from setup_parser import setup_parser
from kg_bert import KGBert_down_zsl_multi_pic
import torch
from trainer import KGBertTrainer_down_zsl_multi_pic
import logging
import glob
from utils import save_best_model, freeze_parameter, unfreeze_parameter
import os

def pad_sequence(batch_data, sentences_ft, masks_ft, batch_token_types, batch_token2fcls, visible_matrixs):
    # Make all tensor in a batch the same length by padding with zeros
    max_len = 0
    for item in batch_data: 
        max_len = max(max_len, len(item))
    
    mask = torch.zeros((len(batch_data),max_len))

    if visible_matrixs is not None:
        final_visible_matrix = torch.zeros((len(batch_data), max_len, max_len))
        for index, item in enumerate(batch_data):
            mask[index][0:len(item)] = 1
            pad_length = max_len-len(item)
            batch_data[index] = batch_data[index] + [config.tokenizer.token2id['[PAD]']]*pad_length
            sentences_ft[index] = sentences_ft[index] + [config.tokenizer.token2id['[PAD]']]*pad_length
            masks_ft[index] = masks_ft[index] + [0] * pad_length
            batch_token_types[index] = batch_token_types[index] + [2] * pad_length 
            
            visible_matrix_len=visible_matrixs[index].shape[0]
            final_visible_matrix[index][0:visible_matrix_len,0:visible_matrix_len] = visible_matrixs[index]

            batch_token2fcls[index] = batch_token2fcls[index] + [-1]*pad_length

    else:
        for index, item in enumerate(batch_data):
            mask[index][0:len(item)] = 1
            pad_length = max_len-len(item)
            
            batch_data[index] = batch_data[index] + [config.tokenizer.token2id['[PAD]']]*pad_length
            sentences_ft[index] = sentences_ft[index] + [config.tokenizer.token2id['[PAD]']]*pad_length
            masks_ft[index] = masks_ft[index] + [0] * pad_length
            batch_token_types[index] = batch_token_types[index] + [2]*pad_length

            final_visible_matrix = None

            batch_token2fcls[index] = batch_token2fcls[index] + [-1]*pad_length


    
    token_type_ids = torch.tensor(batch_token_types)


    batch_data = torch.tensor(batch_data)
    batch_sent_ft = torch.tensor(sentences_ft)
    batch_mask_ft = torch.tensor(masks_ft)
    
    batch_token2fcls = torch.tensor(batch_token2fcls)

    return batch_data, batch_sent_ft, batch_mask_ft, batch_token2fcls, mask.int(), token_type_ids, final_visible_matrix

def collate_fn(batch):
    sentences, sentence_fts, mask_fts, token_types, token2fclses, mask_indexs, labels, visible_matrixs, f_indexs, fids, fcls = [], [], [], [], [], [], [], [], [], [], []
    
    idx = -1
    # Gather in lists, and encode labels as indices
    for sentence_list, sentence_ft_list, mask_ft_list, token_types_list, token2fcls_list, mask_index_list, extended_visible_matrix_list, f_index_list, fid_list, fcls_list, label_list in batch:
        idx += 1
        # if idx % 4 != 0:
        #     continue
        
        sentences += sentence_list
        sentence_fts += sentence_ft_list
        mask_fts += mask_ft_list
        token_types += token_types_list

        token2fclses +=  token2fcls_list
        mask_indexs+=mask_index_list
        visible_matrixs += extended_visible_matrix_list
        f_indexs += f_index_list
        fids += fid_list
        fcls += fcls_list
        labels += label_list

    # Group the list of tensors into a batched tensor
    f_indexs = torch.tensor(f_indexs)
    batch_sentences, batch_sent_ft, batch_mask_ft, batch_token2fcls, attention_mask, token_type_ids, final_visible_matrix = pad_sequence(sentences, sentence_fts, mask_fts, token_types,  token2fclses, visible_matrixs)
    return batch_sentences, batch_sent_ft, batch_mask_ft, batch_token2fcls, attention_mask, token_type_ids, final_visible_matrix, \
        torch.tensor(labels), torch.tensor(mask_indexs), f_indexs, torch.tensor(fids), torch.tensor(fcls)

def collate_fn_test(batch):
    sentences, sentence_fts, mask_fts, token_types, token2fclses, mask_indexs, labels, visible_matrixs, f_indexs, fids, fcls = [], [], [], [], [], [], [], [], [], [], []
    
    # Gather in lists, and encode labels as indices
    for sentence_list, sentence_ft_list, mask_ft_list, token_types_list, token2fcls_list, mask_index_list, extended_visible_matrix_list, f_index_list, fid_list, fcls_list, label_list in batch:
        sentences += sentence_list
        sentence_fts += sentence_ft_list
        mask_fts += mask_ft_list
        token_types += token_types_list

        token2fclses +=  token2fcls_list
        mask_indexs+=mask_index_list
        visible_matrixs += extended_visible_matrix_list
        f_indexs += f_index_list
        fids += fid_list
        fcls += fcls_list
        labels += label_list

    # Group the list of tensors into a batched tensor
    f_indexs = torch.tensor(f_indexs)
    batch_sentences, batch_sent_ft, batch_mask_ft, batch_token2fcls, attention_mask, token_type_ids, final_visible_matrix = pad_sequence(sentences, sentence_fts, mask_fts, token_types, token2fclses, visible_matrixs)
    
    return batch_sentences, batch_sent_ft, batch_mask_ft, batch_token2fcls, attention_mask, token_type_ids, final_visible_matrix, \
        torch.tensor(labels), torch.tensor(mask_indexs), f_indexs, torch.tensor(fids), torch.tensor(fcls)


# ------------------------------------
# setup parser
# ------------------------------------

args = setup_parser()
tokenizer = KGTokenizer(args)
config = Config(tokenizer)

# ------------------------------------
# logging
# ------------------------------------

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename=args.log_file_down_task,
                    filemode='a')

console = logging.StreamHandler()
console.setLevel(logging.INFO)

formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)

logging.getLogger('').addHandler(console)
 
logger = logging.getLogger('logger')
# ------------------------------------
# process data
# ------------------------------------
tokenizer.down_data_zsl()
test_dataset_seen = KGDataset_down_zsl_multi_pic(args.seq_len_down, tokenizer, tokenizer.down_data_test_seen_comb[0:], args)
test_dataset_unseen = KGDataset_down_zsl_multi_pic(args.seq_len_down, tokenizer, tokenizer.down_data_test_unseen_comb[0:], args)
test_loader_seen = DataLoader(
        test_dataset_seen,
        batch_size=args.test_bs,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn_test,
        #num_workers=num_workers,
    )

test_loader_unseen = DataLoader(
        test_dataset_unseen,
        batch_size=args.test_bs,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn_test,
        #num_workers=num_workers,
    )

def ini_train_dataloader(tokenizer):
    tokenizer.down_data_zsl()
    
    train_dataset1 = KGDataset_down_zsl_multi_pic(args.seq_len_down, tokenizer, tokenizer.down_data_train_comb[0:], args, if_fixed=False)
    train_dataset2= KGDataset_down_zsl_multi_pic(args.seq_len_down, tokenizer, tokenizer.down_data_train_comb[0:], args, if_fixed=False)
    
    train_loader1 = DataLoader(
        train_dataset1,
        batch_size=args.train_bs,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_fn,
        #num_workers=num_workers,
    )
    train_loader2 = DataLoader(
        train_dataset2,
        batch_size=args.train_bs,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_fn,
        #num_workers=num_workers,
    )

    
    return train_loader1, train_loader2

# ------------------------------------
# init model and load parameters
# ------------------------------------
logger.info("init KGBert")
KGModel = KGBert_down_zsl_multi_pic(tokenizer,args)

train_loader1, train_loader2 = ini_train_dataloader(tokenizer)
trainer = KGBertTrainer_down_zsl_multi_pic(KGModel, args, logger, tokenizer, train_dataloader=train_loader1,train_dataloader2=train_loader2, test_dataloader=test_loader_unseen, test_dataloader2 = test_loader_seen,cuda_devices=args.cuda_devices, log_freq=args.log_freq)


assert KGModel.encoder.embeddings.word_embeddings.weight.requires_grad == True 
logger.info(f"KGModel.encoder.embeddings.word_embeddings.weight.requires_grad == {KGModel.encoder.embeddings.word_embeddings.weight.requires_grad}")

if args.direct_ft:
    logger.info(f"Directly ft, no pretrained parameters.")
else:
    try:
        parameter_path = args.petrain_save_path + '.ep4_ZSL'
        concept_dict = torch.load(parameter_path)
        KGModel.load_state_dict(concept_dict, strict=False) 
        logger.info(f"load pretrained parameters from {parameter_path}.")
    except:
        logger.info(f"cannot load pretrained parameters.")

if args.fixedT:
    freeze_parameter('encoder.encoder.', KGModel, logger)


# ------------------------------------
# train model
# ------------------------------------
logger.info("Creating BERT Trainer")

logger.info("Training Start")
metric_type = 'T1'
# metric_type = 'S'
last_best_metric = 0 # (acc_all, acc_unseen, acc_seen)
last_best_epoch = -1

def load_best_model():
    parameter_paths = list(glob.iglob(args.down_task_model_path + '.ep*_'+metric_type+'-*'))
    models_max = max([float(i.split(metric_type+'-')[-1]) for i in parameter_paths])
    
    for each_path in parameter_paths:
        if metric_type+'-'+str(models_max) in each_path:
            parameter_path = each_path
    
    print(f'load from {parameter_path}')
    parameter_dict = torch.load(parameter_path)
    try:
        KGModel.load_state_dict(parameter_dict, strict=False) 
    except Exception as e:
        print(e)
        KGModel.load_state_dict(parameter_dict.state_dict(), strict=False) 
    logger.info(f"Load best parameters from {parameter_path}.")

def test_current(epoch, metric_type, trainer):
    if metric_type == 'T1':
        trainer.test2(epoch, args.down_task_model_path) 
        trainer.test(epoch, args.down_task_model_path) 
        last_best_metric = trainer.current_metric[1] 
        
    elif metric_type == 'S': 
        trainer.test(epoch, args.down_task_model_path) 
        trainer.test2(epoch, args.down_task_model_path) 
        last_best_metric = trainer.current_metric[2] 
    
    logger.info(f'test epoch={epoch}, now_best_{metric_type}={last_best_metric}') 
    
    return last_best_metric


if args.continue_pretrain:
    logger.info(f"Load pretrained parameters and continue train.")
    load_best_model()
    

for epoch in range(args.epochs): 
    train_loader1, train_loader2 = ini_train_dataloader(tokenizer)
    trainer = KGBertTrainer_down_zsl_multi_pic(KGModel, args, logger, tokenizer, train_dataloader=train_loader1,train_dataloader2=train_loader2, test_dataloader=test_loader_unseen, test_dataloader2 = test_loader_seen,cuda_devices=args.cuda_devices, log_freq=args.log_freq)

    trainer.train(epoch, args.down_task_model_path) 
    
    if (epoch+1) % args.test_epoch == 0: 
        
        now_metric = test_current(epoch, metric_type, trainer)        

        if now_metric > last_best_metric: 
            logger.info(f"Epoch {epoch}: current_test_metric={now_metric}, better than last_best={last_best_metric}, update model.")
            save_best_model(file_save_path=args.down_task_model_path, logger=logger, metric=metric_type) 
            trainer.save(epoch, args.down_task_model_path, metric=metric_type, value=now_metric)
            last_best_metric = now_metric
            last_best_epoch = epoch
        else:
            logger.info(f"Epoch {epoch}: current_test_metric={now_metric}, not better than last_best={last_best_metric}.")
            if args.lr > 1e-6:
                args.lr = args.lr*0.5
        
    if epoch-last_best_metric>10:
        break