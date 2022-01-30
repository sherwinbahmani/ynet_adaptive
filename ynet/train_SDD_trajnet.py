import pandas as pd
import os
import yaml
from datetime import datetime
from model import YNet
from utils.preprocessing import split_df_ratio, set_random_seeds

FOLDERNAME = './'
seed = 1
set_random_seeds(seed)
time_stamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
CHECKPOINT = None #FOLDERNAME + 'pretrained_models/2022_01_27_23_58_00_weights.pt' # None means no checkpoint will be used to fine-tune
CONFIG_FILE_PATH = 'config/sdd_raw_train.yaml'
DATASET_NAME = 'sdd'
OBS_LEN = 8  # in timesteps
PRED_LEN = 12  # in timesteps
NUM_GOALS = 20  # K_e
NUM_TRAJ = 1  # K_a
BATCH_SIZE = 4
EPOCHS_CHECKPOINTS = 10 # Save checkpoint after every N epochs on top of storing the best one
VAL_RATIO = 0.3 # Take subset of training sets for validation sets (between 0.0 (none) and 1.0 (all))

with open(CONFIG_FILE_PATH) as file:
    params = yaml.load(file, Loader=yaml.FullLoader)    
TRAIN_IMAGE_PATH = FOLDERNAME + 'sdd_raw/annotations'
VAL_IMAGE_PATH = FOLDERNAME + 'sdd_raw/annotations'

## Set up data
DATASET_TYPE = "dataset_ped" # Either dataset_ped_biker, dataset_ped, dataset_biker
GAP_TYPE = "gap" # gap or no_gap
# Gap Range: ["0.25_0.75.pkl", "1.25_1.75.pkl", "2.25_2.75.pkl", "3.25_3.75.pkl"]
DATA_PATH = FOLDERNAME + f'{os.path.join(DATASET_TYPE, GAP_TYPE)}' # either dataset_ped_biker
TRAIN_FILES = ["0.25_0.75.pkl", "1.25_1.75.pkl", "3.25_3.75.pkl"]
VAL_FILES = ["0.25_0.75.pkl", "1.25_1.75.pkl", "3.25_3.75.pkl"]

# No Gap Range: ["0.5_1.5.pkl", "1.5_2.5.pkl", "2.5_3.5.pkl", "3.5_4.5.pkl"]
# DATA_PATH = FOLDERNAME + f'{os.path.join(DATASET_TYPE, GAP_TYPE)}'
# TRAIN_FILES = ["0.5_1.5.pkl", "1.5_2.5.pkl", "3.5_4.5.pkl"]
# VAL_FILES = ["0.5_1.5.pkl", "1.5_2.5.pkl", "3.5_4.5.pkl"]

df_train = pd.concat([pd.read_pickle(os.path.join(DATA_PATH, train_file)) for train_file in TRAIN_FILES])
if TRAIN_FILES == VAL_FILES:
    print(f"Split training set based on given ratio {VAL_RATIO}")
    df_train, df_val = split_df_ratio(df_train, VAL_RATIO)
else:
    df_val = pd.concat([pd.read_pickle(os.path.join(DATA_PATH, val_file)) for val_file in VAL_FILES])
    
params['segmentation_model_fp'] = FOLDERNAME + 'ynet_additional_files/segmentation_models/SDD_segmentation.pth'

model = YNet(obs_len=OBS_LEN, pred_len=PRED_LEN, params=params)
if CHECKPOINT is not None:
    model.load(CHECKPOINT)
    print(f"Loaded checkpoint {CHECKPOINT}")

EXPERIMENT_NAME = ""
EXPERIMENT_NAME += f"Seed_{seed}"
EXPERIMENT_NAME += f"_Train_{'_'.join(['('+f.split('.pkl')[0]+')' for f in TRAIN_FILES])}"
EXPERIMENT_NAME += f"_Val_{'_'.join(['('+f.split('.pkl')[0]+')' for f in VAL_FILES])}"
EXPERIMENT_NAME += f"_Val_Ratio_{VAL_RATIO}"
EXPERIMENT_NAME += f"_{DATASET_TYPE}"
EXPERIMENT_NAME += f"_{GAP_TYPE}"
print(f"Experiment {EXPERIMENT_NAME} has started")


model.train(df_train, df_val, params, train_image_path=TRAIN_IMAGE_PATH, val_image_path=VAL_IMAGE_PATH,
            experiment_name=EXPERIMENT_NAME, batch_size=BATCH_SIZE, num_goals=NUM_GOALS, num_traj=NUM_TRAJ, 
            device=None, dataset_name=DATASET_NAME, use_raw_data=params['use_raw_data'],
            epochs_checkpoints=EPOCHS_CHECKPOINTS)