import pandas as pd
import yaml
import argparse
import torch
from model import YNet
from utils.preprocessing import load_and_window_SDD_small, load_SDD_small

# Custom block
USE_RAW_SMALL = True # Read from raw dataset instead of pickle
SDD_SMALL_PATH = "/fastdata/vilab07/sdd/sdd_small"
# Labels list: ['Biker', 'Bus', 'Car', 'Cart', 'Pedestrian', 'Skater']
TRAIN_LABELS = ['Pedestrian']
TEST_LABELS = []
STEP = 12 #STEP = 12 (2.5 FPS) STEP = 30 () 1 FPS
MIN_NUM_STEPS_SEQ = 8
STRIDE = 8+12 #timesteps to move from one trajectory to the next one

# Orig block
CONFIG_FILE_PATH = 'config/sdd_trajnet.yaml'  # yaml config file containing all the hyperparameters
EXPERIMENT_NAME = 'sdd_trajnet'  # arbitrary name for this experiment
DATASET_NAME = 'sdd'
TRAIN_DATA_PATH = '/fastdata/vilab07/sdd/ynet_additional_files/data/SDD/train_trajnet.pkl'
TRAIN_IMAGE_PATH = '/fastdata/vilab07/sdd/ynet_additional_files/data/SDD/train'
VAL_DATA_PATH = '/fastdata/vilab07/sdd/ynet_additional_files/data/SDD/test_trajnet.pkl'
VAL_IMAGE_PATH = '/fastdata/vilab07/sdd/ynet_additional_files/data/SDD/test'
OBS_LEN = 8  # in timesteps
PRED_LEN = 12  # in timesteps
NUM_GOALS = 20  # K_e
NUM_TRAJ = 1  # K_a

BATCH_SIZE = 4

with open(CONFIG_FILE_PATH) as file:
    params = yaml.load(file, Loader=yaml.FullLoader)
experiment_name = CONFIG_FILE_PATH.split('.yaml')[0].split('config/')[1]


if USE_RAW_SMALL:
    # df_train = load_SDD_small(path=SDD_SMALL_PATH)
    df_train, df_val = load_and_window_SDD_small(path=SDD_SMALL_PATH, step=STEP,
                                                 window_size=MIN_NUM_STEPS_SEQ, stride=STRIDE,
                                                 train_labels=TRAIN_LABELS, test_labels=TEST_LABELS)
else:
    df_train = pd.read_pickle(TRAIN_DATA_PATH)
    df_val = pd.read_pickle(VAL_DATA_PATH)

model = YNet(obs_len=OBS_LEN, pred_len=PRED_LEN, params=params)

model.train(df_train, df_val, params, train_image_path=TRAIN_IMAGE_PATH, val_image_path=VAL_IMAGE_PATH,
            experiment_name=EXPERIMENT_NAME, batch_size=BATCH_SIZE, num_goals=NUM_GOALS, num_traj=NUM_TRAJ, 
            device=None, dataset_name=DATASET_NAME)