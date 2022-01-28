import pandas as pd
import yaml
import argparse
import torch
from model import YNet
from utils.preprocessing import load_and_window_SDD_small, load_SDD_small

FOLDERNAME = './'

# Custom block
USE_RAW_SMALL = False # Read from raw dataset instead of pickle
SDD_SMALL_PATH = FOLDERNAME + "sdd/sdd_small"
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
TRAIN_DATA_PATH = FOLDERNAME + 'data/SDD/train_trajnet.pkl'
TRAIN_IMAGE_PATH = FOLDERNAME + 'data/SDD/train'
VAL_DATA_PATH = FOLDERNAME + 'data/SDD/test_trajnet.pkl'
VAL_IMAGE_PATH = FOLDERNAME + 'data/SDD/test'
OBS_LEN = 8  # in timesteps
PRED_LEN = 12  # in timesteps
NUM_GOALS = 20  # K_e
NUM_TRAJ = 1  # K_a

BATCH_SIZE = 4		# TODO: tuning

with open(CONFIG_FILE_PATH) as file:
	params = yaml.load(file, Loader=yaml.FullLoader)
experiment_name = CONFIG_FILE_PATH.split('.yaml')[0].split('config/')[1]

train_data = pd.read_pickle(TRAIN_DATA_PATH)
val_data = pd.read_pickle(VAL_DATA_PATH)

####################################

def build_model(params):

	ynet = YNet(obs_len=OBS_LEN, pred_len=PRED_LEN, params=params)

	# load pretrained param from the content encoder
	prefix = FOLDERNAME + 'ynet_additional_files'
	state_dict = torch.load(f'{prefix}/pretrained_models/{experiment_name}_weights.pt')
	info = ynet.model.load_state_dict(state_dict, strict=False)
	print("\n\n Load core model: \n", info)

	# initialize style encoder
	enc_dict = {}
	for k, v in state_dict.items():
		if k[:7] == "encoder":
			k = k.replace("encoder.", "")
			enc_dict[k] = v

	# print('enc_dict', enc_dict.keys())
	# print('ynet.model.style_enc', ynet.model.style_enc)

	info = ynet.model.style_enc.load_state_dict(enc_dict, strict=True)
	print("\n\n Init style encoder: \n", info)

	return ynet


def main():
	ynet = build_model(params)

	# # evaluate
	# ynet.evaluate(val_data, params, image_path=VAL_IMAGE_PATH,
	# 			   batch_size=48, rounds=1, 
	# 			   num_goals=NUM_GOALS, num_traj=NUM_TRAJ, device=None, dataset_name=DATASET_NAME)

	# # pre train
	# ynet.train(train_data, val_data, params, train_image_path=TRAIN_IMAGE_PATH, val_image_path=VAL_IMAGE_PATH,
	# 			experiment_name=EXPERIMENT_NAME, batch_size=BATCH_SIZE, num_goals=NUM_GOALS, num_traj=NUM_TRAJ, 
	# 			device=None, dataset_name=DATASET_NAME)

	# sytle enc
	ynet.train_style_enc(train_data, val_data, params, train_image_path=TRAIN_IMAGE_PATH, val_image_path=VAL_IMAGE_PATH,
				experiment_name=EXPERIMENT_NAME, batch_size=BATCH_SIZE, num_goals=NUM_GOALS, num_traj=NUM_TRAJ, 
				device=None, dataset_name=DATASET_NAME)

if __name__ == "__main__":
	main()
