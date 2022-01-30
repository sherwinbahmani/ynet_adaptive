import pandas as pd
import os
import yaml
from datetime import datetime
from model import YNet

# FOLDERNAME = './'
FOLDERNAME = "/fastdata/vilab07/sdd/" 
time_stamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
EXPERIMENT_NAME = time_stamp  # arbitrary name for this experiment
CHECKPOINT = None #FOLDERNAME + 'pretrained_models/2022_01_27_23_58_00_weights.pt' # None means no checkpoint will be used to fine-tune
CONFIG_FILE_PATH = 'config/sdd_raw_train.yaml'
DATASET_NAME = 'sdd'
OBS_LEN = 8  # in timesteps
PRED_LEN = 12  # in timesteps
NUM_GOALS = 20  # K_e
NUM_TRAJ = 1  # K_a
BATCH_SIZE = 4
EPOCHS_CHECKPOINTS = 5 # Save checkpoint after every N epochs on top of storing the best one
print(f"Experiment {EXPERIMENT_NAME} has started")

with open(CONFIG_FILE_PATH) as file:
    params = yaml.load(file, Loader=yaml.FullLoader)    
TRAIN_IMAGE_PATH = FOLDERNAME + 'sdd_raw/annotations'
VAL_IMAGE_PATH = FOLDERNAME + 'sdd_raw/annotations'

## Set up data
DATASET_TYPE = "dataset_ped_biker" # Either dataset_ped_biker, dataset_ped, dataset_biker
# Gap Range: ["0.25_0.75.pkl", "1.25_1.75.pkl", "2.25_2.75.pkl", "3.25_3.75.pkl"]
DATA_PATH = FOLDERNAME + f'{DATASET_TYPE}/gap' # either dataset_ped_biker
TRAIN_FILES = ["0.25_0.75.pkl", "1.25_1.75.pkl", "3.25_3.75.pkl"]
VAL_FILES = ["2.25_2.75.pkl"]

# No Gap Range: ["0.5_1.5.pkl", "1.5_2.5.pkl", "2.5_3.5.pkl", "3.5_4.5.pkl"]
# DATA_PATH = FOLDERNAME + f'{DATASET_TYPE}/no_gap'
# TRAIN_FILES = ["0.5_1.5.pkl", "1.5_2.5.pkl", "3.5_4.5.pkl"]
# VAL_FILES = ["2.5_3.5.pkl"]

df_train = pd.concat([pd.read_pickle(os.path.join(DATA_PATH, train_file)) for train_file in TRAIN_FILES])
df_val = pd.concat([pd.read_pickle(os.path.join(DATA_PATH, val_file)) for val_file in VAL_FILES])
    
params['segmentation_model_fp'] = FOLDERNAME + 'ynet_additional_files/segmentation_models/SDD_segmentation.pth'

model = YNet(obs_len=OBS_LEN, pred_len=PRED_LEN, params=params)
if CHECKPOINT is not None:
    model.load(CHECKPOINT)
    print(f"Loaded checkpoint {CHECKPOINT}")

model.train(df_train, df_val, params, train_image_path=TRAIN_IMAGE_PATH, val_image_path=VAL_IMAGE_PATH,
            experiment_name=EXPERIMENT_NAME, batch_size=BATCH_SIZE, num_goals=NUM_GOALS, num_traj=NUM_TRAJ, 
            device=None, dataset_name=DATASET_NAME, use_raw_data=params['use_raw_data'],
            epochs_checkpoints=EPOCHS_CHECKPOINTS)