from ast import arg
from tkinter import N

from yaml import load
from torch.utils.data import DataLoader
from data import KGDataset_down_qa, KGTokenizer, KGDataset, Config
from setup_parser import setup_parser
from kg_bert import KGBert, KGBert_down_qa
import torch
from trainer import KGBertTrainer_down_qa
import logging
from utils import save_best_model, freeze_parameter, unfreeze_parameter
import glob
import os

def pad_sequence(batch_data, sentences_ft, sentences_ent, masks_ft, batch_token_types, visible_matrixs = None): 
    # Make all tensor in a batch the same length by padding with zeros
    max_len = 0
    for item in batch_data:
        max_len = max(max_len, len(item))

    mask = torch.zeros((len(batch_data), max_len))

    if visible_matrixs is not None:
        final_visible_matrix = torch.zeros((len(batch_data), max_len, max_len))
        for index, item in enumerate(batch_data):
            mask[index][0:len(item)] = 1
            pad_length = max_len-len(item)
            batch_data[index] = batch_data[index] + [config.tokenizer.token2id['[PAD]']]*pad_length
            sentences_ft[index] = sentences_ft[index] + [config.tokenizer.token2id['[PAD]']]*pad_length
            sentences_ent[index] = sentences_ent[index] + [config.tokenizer.token2id['[PAD]']]*pad_length
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
            sentences_ent[index] = sentences_ent[index] + [config.tokenizer.token2id['[PAD]']]*pad_length
            masks_ft[index] = masks_ft[index] + [0] * pad_length
            batch_token_types[index] = batch_token_types[index] + [0]*pad_length

            final_visible_matrix = None


    
    token_type_ids = torch.tensor(batch_token_types)


    batch_data = torch.tensor(batch_data)
    batch_sent_ft = torch.tensor(sentences_ft)
    batch_sent_ent = torch.tensor(sentences_ent)
    batch_mask_ft = torch.tensor(masks_ft)
    

    return batch_data, batch_sent_ft, batch_sent_ent, batch_mask_ft, mask.int(), token_type_ids, final_visible_matrix

def collate_fn_train(batch):
    batch_sentence_list, batch_sentence_ft_list, batch_sentence_ent_list, batch_mask_ft_list, batch_token_types_list, batch_visible_matrix_list, \
        batch_mask_index_list, batch_f_index_list, batch_label, batch_encoder_data = [], [], [], [], [], [], [], [], [], []
    idx = -1
    for sentence_list, sentence_ft_list, sentence_ent_list, mask_ft_list, token_types_list, mask_index_list, visible_matrix_list, f_index_list, \
                    label, encoder_data in batch:
        idx += 1
        if idx % args.train_split != 0: 
            continue
        
        batch_sentence_list += sentence_list
        batch_sentence_ft_list += sentence_ft_list
        batch_sentence_ent_list += sentence_ent_list
        batch_mask_ft_list += mask_ft_list
        batch_token_types_list += token_types_list
        batch_visible_matrix_list += visible_matrix_list
        batch_mask_index_list += mask_index_list
        batch_f_index_list += f_index_list
        batch_label.append(label)
        batch_encoder_data.append(encoder_data)

    batch_sentences, batch_sent_ft, batch_sent_ent, batch_mask_ft, attention_mask, token_type_ids, final_visible_matrix \
        = pad_sequence(batch_sentence_list, batch_sentence_ft_list, batch_sentence_ent_list, batch_mask_ft_list, batch_token_types_list, batch_visible_matrix_list)
    
    return batch_sentences, batch_sent_ft, batch_sent_ent, batch_mask_ft, attention_mask, token_type_ids, final_visible_matrix, \
            torch.tensor(batch_mask_index_list), torch.tensor(batch_f_index_list), \
            torch.tensor(batch_label), batch_encoder_data

def collate_fn(batch):
    batch_sentence_list, batch_sentence_ft_list, batch_sentence_ent_list, batch_mask_ft_list, batch_token_types_list, batch_visible_matrix_list, \
        batch_mask_index_list, batch_f_index_list, batch_label, batch_encoder_data = [], [], [], [], [], [], [], [], [], []
    for sentence_list, sentence_ft_list, sentence_ent_list, mask_ft_list, token_types_list, mask_index_list, visible_matrix_list, f_index_list, \
                    label, encoder_data in batch:
        batch_sentence_list += sentence_list
        batch_sentence_ft_list += sentence_ft_list
        batch_sentence_ent_list += sentence_ent_list
        batch_mask_ft_list += mask_ft_list
        batch_token_types_list += token_types_list
        batch_visible_matrix_list += visible_matrix_list
        batch_mask_index_list += mask_index_list
        batch_f_index_list += f_index_list
        batch_label.append(label)
        batch_encoder_data.append(encoder_data)

    batch_sentences, batch_sent_ft, batch_sent_ent, batch_mask_ft, attention_mask, token_type_ids, final_visible_matrix \
        = pad_sequence(batch_sentence_list, batch_sentence_ft_list, batch_sentence_ent_list, batch_mask_ft_list, batch_token_types_list, batch_visible_matrix_list)
    
    return batch_sentences, batch_sent_ft, batch_sent_ent, batch_mask_ft, attention_mask, token_type_ids, final_visible_matrix, \
            torch.tensor(batch_mask_index_list), torch.tensor(batch_f_index_list), \
            torch.tensor(batch_label), batch_encoder_data


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
tokenizer.down_data_qa()

train_dataset = KGDataset_down_qa(tokenizer, tokenizer.num_choice, tokenizer.train_tri_list, tokenizer.train_qids, tokenizer.train_labels, *tokenizer.train_encoder_data)
valid_dataset = KGDataset_down_qa(tokenizer, tokenizer.num_choice, tokenizer.dev_tri_list, tokenizer.dev_qids, tokenizer.dev_labels, *tokenizer.dev_encoder_data)
test_dataset = KGDataset_down_qa(tokenizer, tokenizer.num_choice, tokenizer.test_tri_list, tokenizer.test_qids, tokenizer.test_labels, *tokenizer.test_encoder_data)


train_loader = DataLoader(
    train_dataset,
    batch_size=args.train_bs*args.train_split,
    shuffle=True,
    drop_last=False,
    collate_fn=collate_fn_train,
    #num_workers=num_workers,
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=args.test_bs,#args.train_bs*args.train_split,#args.test_bs,
    shuffle=False,
    drop_last=False,
    collate_fn=collate_fn,
    #num_workers=num_workers,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=args.test_bs,#args.train_bs*args.train_split,#args.test_bs,
    shuffle=False,
    drop_last=False,
    collate_fn=collate_fn,
    #num_workers=num_workers,
)


KGModel = KGBert_down_qa(tokenizer, args)


if args.direct_ft:
    logger.info(f"Directly ft, no pretrained parameters.")
else:
    parameter_path = args.petrain_save_path + '.ep4_QA'
    concept_dict = torch.load(parameter_path)
    
    for key in list(concept_dict.keys()):
        if 'trans_encoder.encoder.encoder' not in key and 'trans_encoder.encoder.embeddings.word_mlp' not in key:
            del concept_dict[key] 
    
    for name, p in KGModel.named_parameters():
        if name.startswith('trans_encoder.encoder.embeddings.word_mlp.weight'):
            concept_dict['trans_encoder.encoder.embeddings.word_mlp.weight'] += p.data 

    KGModel.load_state_dict(concept_dict, strict=False) 
    for name, p in KGModel.named_parameters():
        if name.startswith('trans_encoder.encoder.encoder.layer'):
            assert (p.data - concept_dict[name]).sum() == 0

    logger.info(f"load pretrained kgtransformer parameters from {parameter_path}.")

# ------------------------------------
# qa ent emb
# ------------------------------------

pretrain_ent_emb = torch.cat((torch.zeros((len(tokenizer.token2id), 1024)),torch.load('dataset/down_qa/ent_emb.pt')),dim=0)
KGModel.trans_encoder.encoder.embeddings.word_embeddings_ent.weight.data.copy_(pretrain_ent_emb)
assert KGModel.trans_encoder.encoder.embeddings.word_embeddings_ent.weight.requires_grad == False 
logger.info(f"KGModel.trans_encoder.encoder.embeddings.word_embeddings_ent.weight.requires_grad == {KGModel.trans_encoder.encoder.embeddings.word_embeddings_ent.weight.requires_grad}")


# ------------------------------------
# trainer
# ------------------------------------

logger.info("Creating BERT Trainer")
trainer = KGBertTrainer_down_qa(KGModel, args, logger, tokenizer, train_dataloader=train_loader,
                                test_dataloader=test_loader, test_dataloader2=valid_loader,
                                with_cuda=True, device0=args.cuda_devices, device1=args.cuda_devices1, log_freq=args.log_freq)


def load_best_model(last_vis=None, current_vis=None, logger = None):
    if not last_vis and not current_vis:
        parameter_paths = list(glob.iglob(args.down_task_model_path + '.ep*_'+metric_type+'-*'))
    else:
        load_from = args.down_task_model_path.replace(current_vis, last_vis)
        parameter_paths = list(glob.iglob(load_from + '.ep*_'+metric_type+'-*'))

    models_max = max([float(i.split(metric_type+'-')[-1]) for i in parameter_paths])
    for each_path in parameter_paths:
        if metric_type+'-'+str(models_max) in each_path:
            parameter_path = each_path
    
    # print(f'load from {parameter_path}')
    parameter_dict = torch.load(parameter_path)

    try:
        KGModel.load_state_dict(parameter_dict, strict=False) 
        logger.info(f"Load best parameters from {parameter_path}.")
    except Exception as e:
        print(parameter_path)
        print(e)
    
last_best_metric = 0 
last_best_epoch = -1
metric_type = 'acc'

if args.fixedT:# fixed KGTransformer ft Roberta
    # trans_encoder.encoder.encoder.
    # freeze_parameter('trans_encoder.encoder.embeddings.word_embeddings', KGModel, logger)
    # freeze_parameter('trans_encoder.encoder.encoder', KGModel, logger) 
    freeze_parameter('trans_encoder.encoder.embeddings.word_mlp', KGModel, logger)
    # freeze_parameter('text_encoder.', KGModel, logger) # if not args.continue_pretrain, need load roberta's best model

if False: 
    if args.train_part == 'roberta': 
        if args.continue_pretrain:
            logger.info(f"Load pretrained parameters and continue train.")
            load_best_model(logger=logger)
    else: 
        if args.continue_pretrain:
            logger.info(f"Load pretrained parameters and continue train.")
            load_best_model(logger=logger)
        else:
            logger.info(f"Load last progress parameters and train.")
            if args.train_part == 'kgtrans': 
                
                load_best_model(last_vis='roberta', current_vis='kgtrans',logger= logger)
            elif args.train_part == 'together':
                load_best_model(last_vis='kgtrans', current_vis='together',logger=logger)
        #last_best_metric = trainer.eval(last_best_epoch, istest=False)
        #trainer.eval(last_best_epoch, istest=True)
    if args.train_part == 'roberta' or args.train_part == "together":
        freeze_epoch = 0 
    elif args.train_part == 'kgtrans':
        freeze_epoch = 1e9

else:# similar to QA-GNN
    if args.continue_pretrain:
        logger.info(f"Load pretrained parameters and continue train.")
        load_best_model(logger=logger)
        # trainer.eval(last_best_epoch)
        # trainer.eval(last_best_epoch, istest=True)
        # 5/0
        # freeze_epoch = 0 

    # else: # Roberta 
        # freeze_epoch = 0
        

# ------------------------------------
# train model
# ------------------------------------

logger.info("Training Start")

for epoch in range(args.epochs):  
    # if epoch < 4: # train Roberta
    #     freeze_parameter('trans_encoder.', KGModel, logger)
    #     unfreeze_parameter('trans_encoder.fc', KGModel, logger)
    # else:# unfreeze trans_encoder
    #     unfreeze_parameter('trans_encoder.', KGModel, logger)
    #     # freeze_parameter('trans_encoder.encoder.encoder', KGModel, logger) 
        # freeze_parameter('trans_encoder.encoder.embeddings.word_mlp', KGModel, logger)

    trainer.train(epoch)
    if (epoch+1)%1 == 0:
        now_metric = trainer.eval(epoch, istest=False)
        trainer.eval(epoch, istest=True)

        if now_metric >= last_best_metric: 
            logger.info(f"Epoch {epoch}: current_test_metric={now_metric}, better than last_best={last_best_metric}, update model.")
            save_best_model(file_save_path=args.down_task_model_path, logger=logger, metric=metric_type, max_num=5) 
            trainer.save(epoch, file_path=args.down_task_model_path, metric=metric_type, value=now_metric)
            last_best_metric = now_metric
            last_best_epoch = epoch
        else:
            logger.info(f"Epoch {epoch}: current_test_metric={now_metric}, not better than last_best={last_best_metric}.")
        
load_best_model(logger=logger)
trainer.eval(last_best_epoch, True)
