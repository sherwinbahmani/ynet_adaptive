import pandas as pd
import os
import yaml
from datetime import datetime
from model import YNet
from utils.preprocessing import split_df_ratio, set_random_seeds

from utils.parser import get_parser

import warnings
warnings.filterwarnings("ignore")

args = get_parser()

print(args)

# TRAIN_FILES = ["0.25_0.75.pkl", "1.25_1.75.pkl", "3.25_3.75.pkl"]
# VAL_FILES = ["0.25_0.75.pkl", "1.25_1.75.pkl", "3.25_3.75.pkl"]

TRAIN_FILES = ["0.25_0.75.pkl", "1.25_1.75.pkl", "2.25_2.75.pkl"]
VAL_FILES = ["0.25_0.75.pkl", "1.25_1.75.pkl", "2.25_2.75.pkl"]

set_random_seeds(args.seed)
time_stamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
CONFIG_FILE_PATH = 'config/sdd_raw_train.yaml'
DATASET_NAME = 'sdd'
OBS_LEN = 8  # in timesteps
PRED_LEN = 12  # in timesteps
NUM_GOALS = 20  # K_e
NUM_TRAJ = 1  # K_a

with open(CONFIG_FILE_PATH) as file:
    params = yaml.load(file, Loader=yaml.FullLoader)    
TRAIN_IMAGE_PATH = args.foldername + 'sdd_raw/annotations'
VAL_IMAGE_PATH = args.foldername + 'sdd_raw/annotations'

assert os.path.isdir(TRAIN_IMAGE_PATH), 'raw data dir error'
assert os.path.isdir(VAL_IMAGE_PATH), 'raw data dir error'

## Set up data
# Gap Range: ["0.25_0.75.pkl", "1.25_1.75.pkl", "2.25_2.75.pkl", "3.25_3.75.pkl"]
DATA_PATH = args.foldername + f'{os.path.join(args.dataset, args.type)}' # either dataset_ped_biker

df_train = pd.concat([pd.read_pickle(os.path.join(DATA_PATH, train_file)) for train_file in TRAIN_FILES])
if TRAIN_FILES == VAL_FILES:
    print(f"Split training set based on given ratio {args.val_ratio}")
    df_train, df_val = split_df_ratio(df_train, args.val_ratio)
else:
    df_val = pd.concat([pd.read_pickle(os.path.join(DATA_PATH, val_file)) for val_file in VAL_FILES])
    
params['segmentation_model_fp'] = args.foldername + 'ynet_additional_files/segmentation_models/SDD_segmentation.pth'

model = YNet(obs_len=OBS_LEN, pred_len=PRED_LEN, params=params)
if args.ckpt is not None:
    model.load(args.ckpt)
    print(f"Loaded checkpoint {args.ckpt}")
else:
	print("Training from scratch")

EXPERIMENT_NAME = ""
EXPERIMENT_NAME += f"Seed_{args.seed}"
EXPERIMENT_NAME += f"_Train_{'_'.join(['('+f.split('.pkl')[0]+')' for f in TRAIN_FILES])}"
EXPERIMENT_NAME += f"_Val_{'_'.join(['('+f.split('.pkl')[0]+')' for f in VAL_FILES])}"
EXPERIMENT_NAME += f"_Val_Ratio_{args.val_ratio}"
EXPERIMENT_NAME += f"_{args.dataset}"
EXPERIMENT_NAME += f"_{args.type}"
print(f"Experiment {EXPERIMENT_NAME} has started")


model.train(df_train, df_val, params, train_image_path=TRAIN_IMAGE_PATH, val_image_path=VAL_IMAGE_PATH,
            experiment_name=EXPERIMENT_NAME, batch_size=args.batch_size, num_goals=NUM_GOALS, num_traj=NUM_TRAJ, 
            device=None, dataset_name=DATASET_NAME, use_raw_data=params['use_raw_data'],
            epochs_checkpoints=args.save_every)