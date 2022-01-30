# mkdir pretrained_models

CKPT='pretrained_models/Seed_1_Train_(0.25_0.75)_(1.25_1.75)_(3.25_3.75)_Val_(0.25_0.75)_(1.25_1.75)_(3.25_3.75)_args.val_ratio_0.3_dataset_ped_gap_weights.pt'

python train_SDD_trajnet.py --dataset='dataset_ped_biker'

# python train_SDD_trajnet.py --dataset='dataset_biker'

# python train_SDD_trajnet.py --dataset='dataset_ped'


# python train_SDD_trajnet.py --batch_size=60 --ckpt='pretrained_models/Seed_1_Train_(0.25_0.75)_(1.25_1.75)_(3.25_3.75)_Val_(0.25_0.75)_(1.25_1.75)_(3.25_3.75)_args.val_ratio_0.3_dataset_ped_gap_weights.pt'