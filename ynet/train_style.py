import pandas as pd
import yaml
import argparse
import torch
from model import YNet, StyleEncoder
from utils.preprocessing import load_and_window_SDD_small, load_SDD_small

# training dependency
from torch.utils.data import DataLoader
from utils.softargmax import SoftArgmax2D, create_meshgrid
from utils.preprocessing import augment_data, create_images_dict
from utils.image_utils import create_gaussian_heatmap_template, create_dist_mat, \
	preprocess_image_for_segmentation, pad, resize
from utils.dataloader import SceneDataset, scene_collate


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

def pre_train(params):
	model = YNet(obs_len=OBS_LEN, pred_len=PRED_LEN, params=params)

	model.train(train_data, val_data, params, train_image_path=TRAIN_IMAGE_PATH, val_image_path=VAL_IMAGE_PATH,
	            experiment_name=EXPERIMENT_NAME, batch_size=BATCH_SIZE, num_goals=NUM_GOALS, num_traj=NUM_TRAJ, 
	            device=None, dataset_name=DATASET_NAME)

####################################

def build_style_enc(params):

	# additional component
	style_enc = StyleEncoder(in_channels=params['semantic_classes'] + OBS_LEN, channels=params['encoder_channels'])

	# load pretrained param from the content encoder
	prefix = FOLDERNAME + 'ynet_additional_files'
	state_dict = torch.load(f'{prefix}/pretrained_models/{experiment_name}_weights.pt')

	enc_dict = {}
	for k, v in state_dict.items():
	    if k[:7] == "encoder":
	        k = k.replace("encoder.", "")
	        enc_dict[k] = v

	style_enc.load_state_dict(enc_dict, strict=True)

	return style_enc

def train_style_enc(encoder, params):

	division_factor = 2 ** len(params['encoder_channels'])

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	obs_len = OBS_LEN
	pred_len = PRED_LEN
	total_len = pred_len + obs_len

	print('Preprocess data')
	image_file_name = 'reference.jpg'

	homo_mat = None
	seg_mask = False

	# Load train images and augment train data and images
	df_train, train_images = augment_data(train_data, image_path=TRAIN_IMAGE_PATH, image_file=image_file_name,
										  seg_mask=seg_mask)

	# Load val scene images
	val_images = create_images_dict(val_data, image_path=VAL_IMAGE_PATH, image_file=image_file_name)

	# Initialize dataloaders
	train_dataset = SceneDataset(df_train, resize=params['resize'], total_len=total_len)
	train_loader = DataLoader(train_dataset, batch_size=1, collate_fn=scene_collate, shuffle=True)

	val_dataset = SceneDataset(val_data, resize=params['resize'], total_len=total_len)
	val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=scene_collate)

	# Preprocess images, in particular resize, pad and normalize as semantic segmentation backbone requires
	resize(train_images, factor=params['resize'], seg_mask=seg_mask)
	pad(train_images, division_factor=division_factor)  # make sure that image shape is divisible by 32, for UNet segmentation
	preprocess_image_for_segmentation(train_images, seg_mask=seg_mask)

	resize(val_images, factor=params['resize'], seg_mask=seg_mask)
	pad(val_images, division_factor=division_factor)  # make sure that image shape is divisible by 32, for UNet segmentation
	preprocess_image_for_segmentation(val_images, seg_mask=seg_mask)

	encoder = encoder.to(device)

	print("TODO: train style encoder")

def main():
	# pre_train(params)

	encoder = build_style_enc(params)
	train_style_enc(encoder, params)

if __name__ == "__main__":
    main()