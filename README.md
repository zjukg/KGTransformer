
# Pretrain
```
python run_pretrain.py --cuda_devices 1 --pretrain_dataset BIG --dataset_name BIG --num_hidden_layers 4 --train_bs 16 --lr 1e-4 --epochs 10 
```

# Downstream Task
## Triple CLS
```
python run_down_triplecls.py --dataset_name WN18RR --pretrain_dataset BIG --down_task down_triplecls --cuda_devices 2 --train_bs 16 --test_bs 128 --epochs 50 --fixedT 1
```
## ZSL
```
python run_down_zsl.py --dataset_name down_zsl --pretrain_dataset down_zsl --down_task down_zsl --cuda_devices 0 --train_bs 32 --test_bs 8 --epochs 10 --fixedT 1 --lr 1e-4 --test_epoch 1 --multi_pic 15
```
## QA 
```
pip install transformers==2.0.0
```

```
python run_down_qa.py --dataset_name down_qa --pretrain_dataset BIG --down_task down_qa --token_types 5 --cuda_devices 0 --cuda_devices1 1  --train_bs 16 --big_bs 64 --train_split 1 --test_bs 64 --epochs 20 --encoder_lr 2e-5 --decoder_lr 1e-4 --fixedT 1 
```


