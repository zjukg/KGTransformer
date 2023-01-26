import torch

pretrain_model = 'pretrain_models/BIG/model_layer-4_hidden-768_heads-12_seq-126_textE-cls_t0-1.0_t1-1.0_t2-1.0.ep4'# pretrain_model

target_file = pretrain_model + '_delWE'

pretrain_weight = torch.load(pretrain_model) 

for key in list(pretrain_weight.keys()):
    
    # if 'trans_encoder.encoder' not in key or 'trans_encoder.encoder.embeddings.word_embeddings_ent.weight' in key:
    #     del pretrain_weight[key] 
    
    if 'encoder.embeddings.token_type_embeddings.weight' in key or'encoder.embeddings.word_embeddings.weight' in key  or 'encoder.embeddings.word_embeddings_ent.weight' in key or 'cls0.linear' in key or 'cls1.linear' in key or 'map_mem' in key:
        del pretrain_weight[key]
        
# import pdb; pdb.set_trace()
torch.save(pretrain_weight,target_file)