import pandas as pd
import pathlib
import os
import yaml
import argparse
import torch
from datetime import datetime
from model import YNet
from utils.preprocessing import load_raw_dataset

# FOLDERNAME = './'
FOLDERNAME = "/fastdata/vilab07/sdd/" 
time_stamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
OUT_PATH_DATA = FOLDERNAME + "dataset_custom"# If None, the datasets will not be saved
EXPERIMENT_NAME = time_stamp  # arbitrary name for this experiment
SDD_RAW_PATH = FOLDERNAME + "sdd_raw"
CHECKPOINT = None #FOLDERNAME + 'pretrained_models/2022_01_27_23_58_00_weights.pt' # None means no checkpoint will be used to fine-tune
# CONFIG_FILE_PATH = 'config/sdd_raw_fine_tune.yaml'  # 'config/sdd_raw_train.yaml' for training from scratch
CONFIG_FILE_PATH = 'config/sdd_raw_train.yaml'
DATASET_NAME = 'sdd'
OBS_LEN = 8  # in timesteps
PRED_LEN = 12  # in timesteps
NUM_GOALS = 20  # K_e
NUM_TRAJ = 1  # K_a
BATCH_SIZE = 4
print(f"Experiment {EXPERIMENT_NAME} has started")

with open(CONFIG_FILE_PATH) as file:
    params = yaml.load(file, Loader=yaml.FullLoader)    
# if params['use_raw_data']: 
#     TRAIN_IMAGE_PATH = FOLDERNAME + 'sdd_raw/annotations'
#     VAL_IMAGE_PATH = FOLDERNAME + 'sdd_raw/annotations'
# else:
#     TRAIN_DATA_PATH = FOLDERNAME + 'ynet_additional_files/data/SDD/train_trajnet.pkl'
#     TRAIN_IMAGE_PATH = FOLDERNAME + 'ynet_additional_files/data/SDD/train'
#     VAL_DATA_PATH = FOLDERNAME + 'ynet_additional_files/data/SDD/test_trajnet.pkl'
#     VAL_IMAGE_PATH = FOLDERNAME + 'ynet_additional_files/data/SDD/test'
TRAIN_IMAGE_PATH = FOLDERNAME + 'sdd_raw/annotations'
VAL_IMAGE_PATH = FOLDERNAME + 'sdd_raw/annotations'
TRAIN_DATA_PATH = FOLDERNAME + 'dataset_custom/2022_01_29_19_13_45_train.pkl'
VAL_DATA_PATH = FOLDERNAME + 'dataset_custom/2022_01_29_19_13_45_val.pkl'
    
params['segmentation_model_fp'] = FOLDERNAME + 'ynet_additional_files/segmentation_models/SDD_segmentation.pth'
# if params['use_raw_data']:
#     df_train, df_val = load_raw_dataset(path=SDD_RAW_PATH, step=params['step'],
#                                                  window_size=params['min_num_steps_seq'], stride=params['filter_stride'],
#                                                  train_labels=params['train_labels'], test_labels=params['test_labels'],
#                                                  test_per=params['test_per'], max_train_agents=params['max_train_agents'],
#                                                  train_set_ratio=params['train_set_ratio'], test_on_train=params['test_on_train'],
#                                                  num_train_agents=params['num_train_agents'], num_test_agents=params['num_test_agents'],
#                                                  random_train_test=params['random_train_test_split'])
#     if OUT_PATH_DATA is not None:
#         pathlib.Path(OUT_PATH_DATA).mkdir(parents=True, exist_ok=True)
#         df_train.to_pickle(os.path.join(OUT_PATH_DATA, f"{time_stamp}_train.pkl"))
#         df_val.to_pickle(os.path.join(OUT_PATH_DATA, f"{time_stamp}_val.pkl"))
    
# else:
#     df_train = pd.read_pickle(TRAIN_DATA_PATH)
#     df_val = pd.read_pickle(VAL_DATA_PATH)
df_train = pd.read_pickle(TRAIN_DATA_PATH)
df_val = pd.read_pickle(VAL_DATA_PATH)

model = YNet(obs_len=OBS_LEN, pred_len=PRED_LEN, params=params)
if CHECKPOINT is not None:
    model.load(CHECKPOINT)
    print(f"Loaded checkpoint {CHECKPOINT}")

model.train(df_train, df_val, params, train_image_path=TRAIN_IMAGE_PATH, val_image_path=VAL_IMAGE_PATH,
            experiment_name=EXPERIMENT_NAME, batch_size=BATCH_SIZE, num_goals=NUM_GOALS, num_traj=NUM_TRAJ, 
            device=None, dataset_name=DATASET_NAME, use_raw_data=params['use_raw_data'])