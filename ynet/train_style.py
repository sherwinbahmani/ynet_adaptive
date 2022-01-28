import pandas as pd
import yaml
from model import YNet
from datetime import datetime
from utils.preprocessing import load_raw_dataset
import time

tic = time.time()

FOLDERNAME = './'
time_stamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
CHECKPOINT = FOLDERNAME + 'pretrained_models/2022_01_27_23_58_00_weights.pt' # None means no checkpoint will be used to fine-tune
CONFIG_FILE_PATH = 'config/sdd_raw_fine_tune.yaml'  # yaml config file containing all the hyperparameters
EXPERIMENT_NAME = time_stamp  # arbitrary name for this experiment
DATASET_NAME = 'sdd'
SDD_RAW_PATH = FOLDERNAME + "sdd_raw"
with open(CONFIG_FILE_PATH) as file:
    params = yaml.load(file, Loader=yaml.FullLoader)
print(f"Experiment {EXPERIMENT_NAME} has started")

if params['use_raw_data']:
    TRAIN_IMAGE_PATH = FOLDERNAME + 'sdd_raw/annotations'
    TEST_IMAGE_PATH = FOLDERNAME + 'sdd_raw/annotations'
else:
    TEST_DATA_PATH = FOLDERNAME + 'data/SDD/test_trajnet.pkl'
    TEST_IMAGE_PATH = FOLDERNAME + 'data/SDD/test'  # only needed for YNet, PECNet ignores this value
params['segmentation_model_fp'] = FOLDERNAME + 'ynet_additional_files/segmentation_models/SDD_segmentation.pth'
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

# if __name__ == "__main__":
# 	main()
# ROUNDS = 1  # Y-net is stochastic. How often to evaluate the whole dataset
# BATCH_SIZE = 8

# if params['use_raw_data']:
#     _, df_test = load_raw_dataset(path=SDD_RAW_PATH, step=params['step'],
#                                   window_size=params['min_num_steps_seq'], stride=params['filter_stride'],
#                                   train_labels=params['train_labels'], test_labels=params['test_labels'],
#                                   test_per=params['test_per'], max_train_agents=params['max_train_agents'],
#                                   train_set_ratio=params['train_set_ratio'], test_on_train=params['test_on_train'],
#                                   num_train_agents=params['num_train_agents'], num_test_agents=params['num_test_agents'],
#                                   random_train_test=params['random_train_test_split'])
# else:
#     df_test = pd.read_pickle(TEST_DATA_PATH)

# model = YNet(obs_len=OBS_LEN, pred_len=PRED_LEN, params=params)
# model.load(CHECKPOINT)
# model.evaluate(df_test, params, image_path=TEST_IMAGE_PATH,
#                batch_size=BATCH_SIZE, rounds=ROUNDS, 
#                num_goals=NUM_GOALS, num_traj=NUM_TRAJ, device=None, dataset_name=DATASET_NAME,
#                use_raw_data=params['use_raw_data'])

# toc = time.time()
# print(time.strftime("%Hh%Mm%Ss", time.gmtime(toc - tic)))
