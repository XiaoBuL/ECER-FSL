# [AAAI-2025] ECER-FSL: Envisioning Class Entity Reasoning by Large Language Models for Few-shot Learning

[**Mushui Liu**](https://xiaobul.github.io) · Fangtai Wu · Bozheng Li · Ziqian Lu · Yunlong Yu ✉ · Xi Li 

Zhejiang University

<a href='https://arxiv.org/pdf/2408.12469'><img src='https://img.shields.io/badge/Arxiv-Paper-red'></a>

## Prepare

Please install the torch 1.13.1 and torchvision 0.14.1

Please download the dataset and put it in the correct path

## Get the pretrain weight

We have given our pretrain weight in [miniImageNet](https://huggingface.co/Shui-VL/ECER-FSL), but you can also train your own by performing our pretrain script:

```Python
python pretrain_clip_adapter.py --backbone_class Res12 --lr 0.1 --query 15
```

## Training scripts

For example, to train the 1-shot 5-way model with Res12 backbone on MiniImageNet:

```Python
python train_fsl.py --max_epoch 50 --model_class MultiSem_Bfusion_Adapter --backbone_class Res12 --dataset MiniImageNet --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 15 --eval_query 15 --balance 0.01 --temperature 64 --temperature2 64 --lr 0.00001 --lr_mul 30 --lr_scheduler step --step_size 10 --gamma 0.5 --init_weights ./saves/initialization/miniimagenet/max_acc_sim_mixloss_TextAdapter.pth --save_dir ./else_check --eval_interval 1 --use_euclidean  --gpu 3 --seed 3
```

## Citation
If you find our work, this repository, or pretrained models useful, please consider giving a star :star: and citation.
```bibtex
@inproceedings{liu2025envisioning,
  title={Envisioning class entity reasoning by large language models for few-shot learning},
  author={Liu, Mushui and Wu, Fangtai and Li, Bozheng and Lu, Ziqian and Yu, Yunlong and Li, Xi},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={18},
  pages={18906--18914},
  year={2025}
}
```

## Contact
If you have any questions, please create an issue on this repository or contact at lms@zju.edu.cn.


