# ECER-FSL

# Prepare

Please install the torch 1.13.1 and torchvision 0.14.1

Please download the dataset and put them in the correct path

# Get the pretrain weigth

We have given our pretrain weight in ./saves/initialiazation ,but you can also train your own by performing our pretrain script:

```Python
python pretrain_clip_adapter.py --backbone_class Res12 --lr 0.1 --query 15
```

# Training scripts

For example, to train the 1-shot 5-way model with Res12 backbone on MiniImageNet:

```Python
python train_fsl.py --max_epoch 50 --model_class MultiSem_Bfusion_Adapter --backbone_class Res12 --dataset MiniImageNet --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 15 --eval_query 15 --balance 0.01 --temperature 64 --temperature2 64 --lr 0.00001 --lr_mul 30 --lr_scheduler step --step_size 10 --gamma 0.5 --init_weights ./saves/initialization/miniimagenet/max_acc_sim_mixloss_TextAdapter.pth --save_dir ./else_check --eval_interval 1 --use_euclidean  --gpu 3 --seed 3
```

