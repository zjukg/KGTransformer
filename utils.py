import glob
import os

def save_best_model(file_save_path, logger, max_num=2, metric=None):
    if (metric is not None):
        logger.info(f'save model with best {metric}')
        models_path = list(glob.iglob(file_save_path + '.ep*_'+metric+'-*'))
        models_index = [float(i.split(metric+'-')[-1]) for i in models_path]
        if models_index:
            save_index = max(models_index) 
            for model_path in models_path:
                if metric+'-'+str(save_index) not in model_path:
                    logger.info(f'delete {model_path}')
                    os.remove(model_path)
        
    else: # pretrain
        logger.info(f'save model no more than {max_num}')
        models_path = list(glob.iglob(file_save_path + '.ep*'))
        models_index = [int(i.split('.ep')[-1]) for i in models_path]
        models_index.sort()
        delete_index = models_index[:-max_num]
        for di in delete_index:
            delete_file = file_save_path + '.ep' + str(di)
            logger.info(f'delete {delete_file}')
            os.remove(delete_file)


def unfreeze_parameter(unfixed_para, KGModel, logger):
    for name, p in KGModel.named_parameters():
        if name.startswith(unfixed_para):
            p.requires_grad = True
    logger.info(f'unfreeze parameters of {unfixed_para}')
    
    for name, p in KGModel.named_parameters():
        if name.startswith(unfixed_para):
            assert p.requires_grad == True, 'error'

def freeze_parameter(fixed_para, KGModel, logger):
    for name, p in KGModel.named_parameters():
        if name.startswith(fixed_para):
            p.requires_grad = False
            logger.info(f'freeze parameters of {name}')
    logger.info(f'freeze parameters of {fixed_para}')
    
    for name, p in KGModel.named_parameters():
        if name.startswith(fixed_para):
            assert p.requires_grad == False, 'error'

            