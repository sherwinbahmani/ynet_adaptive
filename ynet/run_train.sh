# mkdir pretrained_models

DATASET=dataset_ped_biker

CKPT=ckpt/Seed_1_Train__0.25_0.75__1.25_1.75__3.25_3.75__Val__0.25_0.75__1.25_1.75__3.25_3.75__Val_Ratio_0.3_dataset_ped_gap_weights.pt

python train_SDD_trajnet.py --dataset=${DATASET} --ckpt=${CKPT}

# python train_SDD_trajnet.py --dataset='dataset_biker'

# python train_SDD_trajnet.py --dataset='dataset_ped'
