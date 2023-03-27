from tkinter import N
from torch.utils.data import DataLoader
from data import KGDataset_down_triplecls, KGTokenizer, KGDataset, Config
from setup_parser import setup_parser
from kg_bert import KGBert, KGBert_down_tirplecls
import torch
from trainer import KGBertTrainer_down_triplecls
import logging
from utils import save_best_model
import glob
import os

def pad_sequence(batch_data, sentences_ft, masks_ft, batch_token_types, batch_label, visible_matrixs = None):
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
        
    else:
        for index, item in enumerate(batch_data):
            mask[index][0:len(item)] = 1
            pad_length = max_len-len(item)
            
            batch_data[index] = batch_data[index] + [config.tokenizer.token2id['[PAD]']]*pad_length
            sentences_ft[index] = sentences_ft[index] + [config.tokenizer.token2id['[PAD]']]*pad_length
            masks_ft[index] = masks_ft[index] + [0] * pad_length
            batch_token_types[index] = batch_token_types[index] + [0]*pad_length

            final_visible_matrix = None

    batch = torch.tensor(batch_data)
    batch_sent_ft = torch.tensor(sentences_ft)
    batch_mask_ft = torch.tensor(masks_ft)
    label = torch.tensor(batch_label)
    token_type_ids = torch.tensor(batch_token_types)
    # import pdb; pdb.set_trace()
    return batch, batch_sent_ft, batch_mask_ft, mask.int(), token_type_ids, label, final_visible_matrix


def collate_fn(batch):
    
    sentences, sentence_fts, mask_fts,  labels, visible_matrixs, task_indexs, token_types = [], [], [], [], [], [], []

    for senten, senten_ft, mask_ft, label, visible_matrix, task_index, token_type in batch:
            sentences += senten
            sentence_fts += senten_ft
            mask_fts += mask_ft
            labels += label
            visible_matrixs += visible_matrix
            task_indexs += task_index
            token_types += token_type

  
    batch_sentences, batch_sent_ft, batch_mask_ft, mask, token_type_ids, batch_labels, final_visible_matrix = pad_sequence(sentences, sentence_fts, mask_fts, token_types, labels, visible_matrixs)
    
    return batch_sentences, batch_sent_ft, batch_mask_ft, mask, token_type_ids, batch_labels, final_visible_matrix, torch.tensor(task_indexs)

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
tokenizer.down_data_triplecls()

train_dataset = KGDataset_down_triplecls(args.seq_len_down, tokenizer, tokenizer.down_data_train[0:], args)
valid_dataset = KGDataset_down_triplecls(args.seq_len_down, tokenizer, tokenizer.down_data_valid[0:], args)
test_dataset = KGDataset_down_triplecls(args.seq_len_down, tokenizer, tokenizer.down_data_test[0:], args)


train_loader = DataLoader(
    train_dataset, 
    batch_size=args.train_bs, 
    shuffle=True, 
    collate_fn=collate_fn,
    #num_workers=num_workers,
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=args.test_bs,
    shuffle=False,
    drop_last=False,
    collate_fn=collate_fn,
    #num_workers=num_workers,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=args.test_bs,
    shuffle=False,
    drop_last=False,
    collate_fn=collate_fn,
    #num_workers=num_workers,
)

# ------------------------------------
# init model and load parameters
# ------------------------------------

KGModel = KGBert_down_tirplecls(tokenizer, args)

assert KGModel.encoder.embeddings.word_embeddings.weight.requires_grad == True 
logger.info(f"KGModel.encoder.embeddings.word_embeddings.weight.requires_grad == {KGModel.encoder.embeddings.word_embeddings.weight.requires_grad}")

if args.direct_ft:
    logger.info(f"Directly ft, no pretrained parameters.")
else:
    try:
        parameter_paths = [int(i.split('.ep')[-1].split('_delWE')[0]) for i in list(glob.iglob(args.petrain_save_path + '.ep*_delWE'))]
        parameter_paths.sort()
        parameter_path = args.petrain_save_path + '.ep' + str(parameter_paths[-1]) + '_delWE'
        concept_dict = torch.load(parameter_path)
        KGModel.load_state_dict(concept_dict, strict=False) 
        logger.info(f"load pretrained parameters from {parameter_path}.")
    except:
        logger.info(f"cannot load pretrained parameters.")

if args.fixedT:
    for name, p in KGModel.named_parameters():
        if 'encoder.encoder.layer.' in name:
            p.requires_grad = False
    logger.info('freeze parameters of encoder.encoder.layer.')
    
    for name, p in KGModel.named_parameters():
        if 'encoder.encoder.layer.' in name:
            assert p.requires_grad == False, 'error'

# import pdb; pdb.set_trace()
# ------------------------------------
# train model
# ------------------------------------

logger.info("Creating BERT Trainer")
test_type = 'test'
if test_type == 'valid':
    trainer = KGBertTrainer_down_triplecls(KGModel, args, logger, tokenizer, train_dataloader=train_loader, test_dataloader=valid_loader,cuda_devices=args.cuda_devices, log_freq=args.log_freq, test_type='valid')
elif test_type == 'test':
    trainer = KGBertTrainer_down_triplecls(KGModel, args, logger, tokenizer, train_dataloader=train_loader, test_dataloader=test_loader,cuda_devices=args.cuda_devices, log_freq=args.log_freq, test_type='test')

logger.info("Training Start")
last_best_metric = 0 
last_best_epoch = -1
metric_type = 'f1'

def test_current(epoch, metric_type):
    trainer.test(epoch, args.down_task_model_path) 
    last_best_metric = trainer.current_metric 
    logger.info(f'test epoch={epoch}, now_best_{metric_type}={last_best_metric}') 
    return last_best_metric

def load_best_model():
    parameter_paths = list(glob.iglob(args.down_task_model_path + '.ep*_'+metric_type+'-*'))
    models_max = max([float(i.split(metric_type+'-')[-1]) for i in parameter_paths])
    for each_path in parameter_paths:
        if metric_type+'-'+str(models_max) in each_path:
            parameter_path = each_path
    
    parameter_dict = torch.load(parameter_path)
    try:
        KGModel.load_state_dict(parameter_dict, strict=False) 
    except Exception as e:
        print(parameter_path)
        print(e)
    logger.info(f"Load best parameters from {parameter_path}.")



if args.continue_pretrain:
    logger.info(f"Load pretrained parameters and continue train.")
    load_best_model()
    last_best_metric = test_current(epoch = last_best_epoch, metric_type=metric_type) 
    

for epoch in range(args.epochs):    
    if epoch - last_best_epoch > 10: 
        break

    # now_metric = test_current(epoch, metric_type)  
    trainer.train(epoch) 
    now_metric = test_current(epoch, metric_type)  

    if now_metric > last_best_metric: 
        logger.info(f"Epoch {epoch}: current_test_metric={now_metric}, better than last_best={last_best_metric}, update model.")
        save_best_model(file_save_path=args.down_task_model_path, logger=logger, metric=metric_type) 
        trainer.save(epoch, args.down_task_model_path, metric=metric_type, value=now_metric)
        last_best_metric = now_metric
        last_best_epoch = epoch
    else:
        logger.info(f"Epoch {epoch}: current_test_metric={now_metric}, not better than last_best={last_best_metric}.")
        
        