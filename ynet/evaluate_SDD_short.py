import pandas as pd
# import pickle5 as pickle
import yaml
import argparse
import torch
from model import YNet
from datetime import datetime
from utils.preprocessing import load_and_window_SDD_small

# Custom block
time_stamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
USE_RAW_SMALL = True # Read from raw dataset instead of pickle
SDD_SMALL_PATH = "/fastdata/vilab07/sdd/sdd_small"
# Labels list: ['Biker', 'Bus', 'Car', 'Cart', 'Pedestrian', 'Skater']
TRAIN_LABELS = ['Skater']
TEST_LABELS = ['Skater']
# Filter dataset
STEP = 12 #STEP = 12 (2.5 FPS) STEP = 30 () 1 FPS
MIN_NUM_STEPS_SEQ = 20
STRIDE = 8+12 # timesteps to move from one trajectory to the next one
TEST_PER = 0.3 # percentage for setting number of testing agents based on number of training agents per class
MAX_TRAIN_AGENTS = 10000 #202 # further constrain the number of training agents per class
TRAIN_SET_RATIO = None #0.8
TEST_ON_TRAIN = True # Instead of splitting train into train and test, test on train to have all data
NUM_TRAIN_AGENTS = 4 * 1
NUM_TEST_AGENTS = 100
RANDOM_TRAIN_TEST_SPLIT = False

# Orig block
# CHECKPOINT = None # None means no checkpoint will be used to fine-tune
CHECKPOINT = '/visinf/home/vilab07/sdd/vita_epfl_causal/ynet/pretrained_models/2022_01_27_23_58_00_weights.pt'
CONFIG_FILE_PATH = 'config/sdd_trajnet.yaml'  # yaml config file containing all the hyperparameters
EXPERIMENT_NAME = time_stamp  # arbitrary name for this experiment
print(f"Experiment {EXPERIMENT_NAME} has started")
DATASET_NAME = 'sdd'
if USE_RAW_SMALL:
    TRAIN_IMAGE_PATH = '/fastdata/vilab07/sdd/sdd_small/annotations'
    TEST_IMAGE_PATH = '/fastdata/vilab07/sdd/sdd_small/annotations'
else:
    TEST_DATA_PATH = '/fastdata/vilab07/sdd/ynet_additional_files/data/SDD/test_trajnet.pkl'
    TEST_IMAGE_PATH = '/fastdata/vilab07/sdd/ynet_additional_files/data/SDD/test'  # only needed for YNet, PECNet ignores this value 
######

OBS_LEN = 8  # in timesteps
PRED_LEN = 12  # in timesteps
NUM_GOALS = 20  # K_e
NUM_TRAJ = 1  # K_a

ROUNDS = 1  # Y-net is stochastic. How often to evaluate the whole dataset
BATCH_SIZE = 8

with open(CONFIG_FILE_PATH) as file:
    params = yaml.load(file, Loader=yaml.FullLoader)
experiment_name = CONFIG_FILE_PATH.split('.yaml')[0].split('config/')[1]

if USE_RAW_SMALL:
    _, df_test = load_and_window_SDD_small(path=SDD_SMALL_PATH, step=STEP,
                                           window_size=MIN_NUM_STEPS_SEQ, stride=STRIDE,
                                           train_labels=TRAIN_LABELS, test_labels=TEST_LABELS,
                                           test_per=TEST_PER, max_train_agents=MAX_TRAIN_AGENTS,
                                           train_set_ratio=TRAIN_SET_RATIO, test_on_train=TEST_ON_TRAIN,
                                           num_train_agents=NUM_TRAIN_AGENTS, num_test_agents=NUM_TEST_AGENTS,
                                           random_train_test=RANDOM_TRAIN_TEST_SPLIT)
else:
    df_test = pd.read_pickle(TEST_DATA_PATH)

model = YNet(obs_len=OBS_LEN, pred_len=PRED_LEN, params=params)
model.load(CHECKPOINT)
model.evaluate(df_test, params, image_path=TEST_IMAGE_PATH,
               batch_size=BATCH_SIZE, rounds=ROUNDS, 
               num_goals=NUM_GOALS, num_traj=NUM_TRAJ, device=None, dataset_name=DATASET_NAME,
               use_raw_small=USE_RAW_SMALL)

