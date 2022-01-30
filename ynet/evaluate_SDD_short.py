import pandas as pd
import yaml
from model import YNet
from datetime import datetime
import time
import os
from utils.preprocessing import set_random_seeds

tic = time.time()

FOLDERNAME = './'
seed = 1
set_random_seeds(seed)
time_stamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
CHECKPOINT = None #FOLDERNAME + 'pretrained_models/2022_01_27_23_58_00_weights.pt' # None means no checkpoint will be used to fine-tune
CONFIG_FILE_PATH = 'config/sdd_raw_eval.yaml'  # yaml config file containing all the hyperparameters
EXPERIMENT_NAME = time_stamp  # arbitrary name for this experiment
DATASET_NAME = 'sdd'
with open(CONFIG_FILE_PATH) as file:
    params = yaml.load(file, Loader=yaml.FullLoader)
print(f"Experiment {EXPERIMENT_NAME} has started")

params['segmentation_model_fp'] = FOLDERNAME + 'ynet_additional_files/segmentation_models/SDD_segmentation.pth'
OBS_LEN = 8  # in timesteps
PRED_LEN = 12  # in timesteps
NUM_GOALS = 20  # K_e
NUM_TRAJ = 1  # K_a
ROUNDS = 1  # Y-net is stochastic. How often to evaluate the whole dataset
BATCH_SIZE = 8

TEST_IMAGE_PATH = FOLDERNAME + 'sdd_raw/annotations'
## Set up data
DATASET_TYPE = "dataset_ped" # Either dataset_ped_biker, dataset_ped, dataset_biker
# Gap Range: ["0.25_0.75.pkl", "1.25_1.75.pkl", "2.25_2.75.pkl", "3.25_3.75.pkl"]
DATA_PATH = FOLDERNAME + f'{DATASET_TYPE}/gap' # either dataset_ped_biker
TEST_FILES = ["2.25_2.75.pkl"]

# No Gap Range: ["0.5_1.5.pkl", "1.5_2.5.pkl", "2.5_3.5.pkl", "3.5_4.5.pkl"]
# DATA_PATH = FOLDERNAME + f'{DATASET_TYPE}/no_gap'
# TEST_FILES = ["2.5_3.5.pkl"]

df_test = pd.concat([pd.read_pickle(os.path.join(DATA_PATH, test_file)) for test_file in TEST_FILES])

model = YNet(obs_len=OBS_LEN, pred_len=PRED_LEN, params=params)
if CHECKPOINT is not None:
    model.load(CHECKPOINT)
    print(f"Loaded checkpoint {CHECKPOINT}")
else:
    raise ValueError("No checkpoint given!")
model.evaluate(df_test, params, image_path=TEST_IMAGE_PATH,
               batch_size=BATCH_SIZE, rounds=ROUNDS, 
               num_goals=NUM_GOALS, num_traj=NUM_TRAJ, device=None, dataset_name=DATASET_NAME,
               use_raw_data=params['use_raw_data'])

toc = time.time()
print(time.strftime("%Hh%Mm%Ss", time.gmtime(toc - tic)))