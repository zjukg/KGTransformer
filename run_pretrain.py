from tkinter import N
from torch.utils.data import DataLoader
from data import KGTokenizer, KGDataset, Config
from setup_parser import setup_parser
from kg_bert import KGBert
import torch
from trainer import KGBertTrainer
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
    # token_type_ids = torch.zeros((len(batch_data),max_len))

    if visible_matrixs is not None:
        final_visible_matrix = torch.zeros((len(batch_data), max_len, max_len))
        for index, item in enumerate(batch_data):
            mask[index][0:len(item)] = 1
            pad_length = max_len-len(item)
            
            batch_data[index] = batch_data[index] + [config.tokenizer.token2id['[PAD]']]*pad_length
            sentences_ft[index] = sentences_ft[index] + [config.tokenizer.token2id['[PAD]']]*pad_length
            masks_ft[index] = masks_ft[index] + [0] * pad_length
            batch_label[index] = batch_label[index] + [-1] * pad_length
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
            batch_label[index] = batch_label[index] + [-1]*pad_length
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
    

    sentences, sentences_ft, masks_ft, types, labels, visible_matrixs, token_types, spe_g_indexs = [], [], [], [], [], [], [], [] 
    # Gather in lists, and encode labels as indices
    for senten0, senten_ft0, mask_ft0, task0, label0, visible_matrix0, token_types0, spe_g_index_0, \
        senten1, senten_ft1, mask_ft1, task1, label1, visible_matrix1, token_types1, \
        senten2, senten_ft2, mask_ft2, task2, label2, visible_matrix2, token_types2 in batch:
        
        sentences.append(senten0)
        sentences_ft.append(senten_ft0)
        masks_ft.append(mask_ft0)
        types.append(task0)
        labels.append(label0)
        visible_matrixs.append(visible_matrix0)
        token_types.append(token_types0)
        spe_g_indexs.append(spe_g_index_0)

        sentences.append(senten1)
        sentences_ft.append(senten_ft1)
        masks_ft.append(mask_ft1)
        types.append(task1)
        labels.append(label1)
        visible_matrixs.append(visible_matrix1)
        token_types.append(token_types1)

        sentences.append(senten2)
        sentences_ft.append(senten_ft2)
        masks_ft.append(mask_ft2)
        types.append(task2)
        labels.append(label2)
        visible_matrixs.append(visible_matrix2)
        token_types.append(token_types2)

    
    
    batch_sentences, batch_sent_ft, batch_mask_ft, mask, token_type_ids, batch_labels, final_visible_matrix = pad_sequence(sentences, sentences_ft, masks_ft, token_types, labels, visible_matrixs)
    
    return batch_sentences, batch_sent_ft, batch_mask_ft, mask, token_type_ids, torch.tensor(types), batch_labels, final_visible_matrix, torch.tensor(spe_g_indexs)

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
                    filename=args.log_file,
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

train_dataset = KGDataset(args.seq_len_pre, tokenizer, tokenizer.train_triples + tokenizer.valid_triples + tokenizer.test_triples)
valid_dataset = KGDataset(args.seq_len_pre, tokenizer, tokenizer.valid_triples) 
test_dataset = KGDataset(args.seq_len_pre, tokenizer, tokenizer.test_triples)


train_loader = DataLoader(
    train_dataset,
    batch_size=args.train_bs,
    shuffle=True,
    drop_last=False,
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


KGModel = KGBert(tokenizer, args)

if args.continue_pretrain:
    logger.info(f"Load pretrained parameters and continue train.")
    parameter_paths = [int(i.split('.ep')[-1]) for i in list(glob.iglob(args.petrain_save_path + '.ep*'))]
    parameter_paths.sort()
    parameter_path = args.petrain_save_path + '.ep' + str(parameter_paths[-1])
    parameter_dict = torch.load(parameter_path)
    try:
        KGModel.load_state_dict(parameter_dict, strict=False) 
    except Exception as e:
        print(e)
        KGModel.load_state_dict(parameter_dict.state_dict(), strict=False) 
    logger.info(f"Load pretrained parameters from {parameter_path}.")

assert KGModel.encoder.embeddings.word_embeddings.weight.requires_grad == True 
logger.info(f"KGModel.encoder.embeddings.word_embeddings.weight.requires_grad == {KGModel.encoder.embeddings.word_embeddings.weight.requires_grad}")
# ------------------------------------
# train model
# ------------------------------------

logger.info("Creating BERT Trainer")
trainer = KGBertTrainer(KGModel, args, logger, tokenizer, train_dataloader=train_loader, test_dataloader=valid_loader,cuda_devices=args.cuda_devices, log_freq=args.log_freq)

logger.info("Training Start")
    
last_loss = 1e9 
for epoch in range(args.epochs):    

    trainer.train(epoch)    
    trainer.test(epoch)

    if trainer.current_metric < last_loss: 
        logger.info(f"current_loss={trainer.current_metric}, less than last_best={last_loss}.")
        save_best_model(args.petrain_save_path, logger=logger, max_num=2)
        trainer.save(epoch, args.petrain_save_path) 
        last_loss = trainer.current_metric

    else:
        logger.info(f"current_loss={trainer.current_metric}, not less than last_best={last_loss}.")
        trainer.save(epoch, args.petrain_save_path) 
    
        

        