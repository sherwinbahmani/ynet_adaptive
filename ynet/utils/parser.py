import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    # general
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--foldername", default='/work/vita/datasets/sdd/', type=str)
    parser.add_argument("--save_every", default=50, type=int)

    # data
    parser.add_argument("--val_ratio", default=0.3, type=float)
    parser.add_argument("--dataset", default='dataset_ped', type=str, help='dataset_ped_biker, dataset_ped, dataset_biker')
    parser.add_argument("--type", default='gap', type=str, help='gap or no_gap')

    # load
    parser.add_argument("--ckpt", default=None, type=str, help='gap or no_gap')

# TRAIN_FILES = ["0.25_0.75.pkl", "1.25_1.75.pkl", "3.25_3.75.pkl"]
# VAL_FILES = ["0.25_0.75.pkl", "1.25_1.75.pkl", "3.25_3.75.pkl"]

    return parser.parse_args()
