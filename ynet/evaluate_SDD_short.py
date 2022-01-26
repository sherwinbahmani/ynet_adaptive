import pandas as pd
# import pickle5 as pickle
import yaml
import argparse
import torch
from model import YNet

CONFIG_FILE_PATH = 'config/sdd_trajnet.yaml'  # yaml config file containing all the hyperparameters
DATASET_NAME = 'sdd'
TEST_DATA_PATH = '/fastdata/vilab07/sdd/ynet_additional_files/data/SDD/test_trajnet.pkl'
TEST_IMAGE_PATH = '/fastdata/vilab07/sdd/ynet_additional_files/data/SDD/test'  # only needed for YNet, PECNet ignores this value
OBS_LEN = 8  # in timesteps
PRED_LEN = 12  # in timesteps
NUM_GOALS = 20  # K_e
NUM_TRAJ = 1  # K_a

ROUNDS = 1  # Y-net is stochastic. How often to evaluate the whole dataset
BATCH_SIZE = 8

with open(CONFIG_FILE_PATH) as file:
    params = yaml.load(file, Loader=yaml.FullLoader)
experiment_name = CONFIG_FILE_PATH.split('.yaml')[0].split('config/')[1]

df_test = pd.read_pickle(TEST_DATA_PATH)
model = YNet(obs_len=OBS_LEN, pred_len=PRED_LEN, params=params)
model.load(f'pretrained_models/{experiment_name}_weights.pt')
model.evaluate(df_test, params, image_path=TEST_IMAGE_PATH,
               batch_size=BATCH_SIZE, rounds=ROUNDS, 
               num_goals=NUM_GOALS, num_traj=NUM_TRAJ, device=None, dataset_name=DATASET_NAME)

