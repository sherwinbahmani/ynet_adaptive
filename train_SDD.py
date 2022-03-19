import pandas as pd
import os
import yaml
from datetime import datetime
import time
from model import YNet
from utils.dataset import set_random_seeds, limit_samples, split_df_ratio

from utils.parser import train_parser
from utils.write_files import write_csv

import warnings
warnings.filterwarnings("ignore")

args = train_parser()

if args.gpu is not None:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

print(args)

tic = time.time()
set_random_seeds(args.seed)
time_stamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
CONFIG_FILE_PATH = os.path.join('config', 'sdd_raw_train.yaml')
DATASET_NAME = 'sdd'

with open(CONFIG_FILE_PATH) as file:
    params = yaml.load(file, Loader=yaml.FullLoader)    
TRAIN_IMAGE_PATH = os.path.join(args.foldername, 'dataset_raw', 'annotations')
VAL_IMAGE_PATH = os.path.join(args.foldername, 'dataset_raw', 'annotations')

# set the learning rate depending on the model
params['learning_rate'] = args.learning_rate
# if args.train_net == 'modulator':
#     params['learning_rate'] = 0.01

assert os.path.isdir(TRAIN_IMAGE_PATH), 'raw data dir error'
assert os.path.isdir(VAL_IMAGE_PATH), 'raw data dir error'

## Set up data
DATA_PATH = os.path.join(args.foldername, args.dataset)
df_train = pd.concat([pd.read_pickle(os.path.join(DATA_PATH, train_file)) for train_file in args.train_files])
if args.train_files == args.val_files:
    print(f"Split training set based on given ratio {args.val_ratio}")
    df_train, df_val = split_df_ratio(df_train, args.val_ratio)
else:
    df_val = pd.concat([pd.read_pickle(os.path.join(DATA_PATH, val_file)) for val_file in args.val_files])

df_train = limit_samples(df_train, args.num_train_batches, args.batch_size)
    
params['segmentation_model_fp'] = os.path.join(args.foldername, 'ynet_additional_files', 'segmentation_models', 'SDD_segmentation.pth')
params['num_epochs'] = args.num_epochs

model = YNet(obs_len=params['OBS_LEN'], pred_len=params['PRED_LEN'], params=params)

if args.train_net == "modulator":
    model.model.initialize_style()

if args.ckpt is not None:
    model.load(args.ckpt)
    print(f"Loaded checkpoint {args.ckpt}")
else:
	print("Training from scratch")

EXPERIMENT_NAME = ""
EXPERIMENT_NAME += f"Seed_{args.seed}"
EXPERIMENT_NAME += f"_Train_{'_'.join(['_'+f.split('.pkl')[0] for f in args.train_files])}_"
EXPERIMENT_NAME += f"_Val_{'_'.join(['_'+f.split('.pkl')[0] for f in args.val_files])}_"
EXPERIMENT_NAME += f"_Val_Ratio_{args.val_ratio}"
EXPERIMENT_NAME += f"_{(args.dataset).replace('/', '_')}"
EXPERIMENT_NAME += f"_train_{args.train_net}"
print(f"Experiment {EXPERIMENT_NAME} has started")


val_ade, val_fde = model.train(df_train, df_val, params, train_image_path=TRAIN_IMAGE_PATH, val_image_path=VAL_IMAGE_PATH,
            experiment_name=EXPERIMENT_NAME, batch_size=args.batch_size, num_goals=params['NUM_GOALS'], num_traj=params['NUM_TRAJ'], 
            device=None, dataset_name=DATASET_NAME, use_raw_data=params['use_raw_data'],
            epochs_checkpoints=args.save_every, train_net=args.train_net, fine_tune=args.fine_tune)

if args.out_csv_dir is not None and args.fine_tune:
    ade_final, fde_final = model.evaluate(df_val, params, image_path=VAL_IMAGE_PATH,
            batch_size=args.batch_size, rounds=args.rounds, 
            num_goals=params['NUM_GOALS'], num_traj=params['NUM_TRAJ'], device=None, dataset_name=DATASET_NAME,
            use_raw_data=params['use_raw_data'], with_style=args.train_net == "modulator")

    num_train_batches = len(df_train)//((params['OBS_LEN'] + params['PRED_LEN']) * args.batch_size)
    write_csv(args.out_csv_dir, args.seed, val_ade, val_fde, args.num_epochs, num_train_batches, args.train_net, args.dataset,
              args.val_files, args.train_files, ade_final, fde_final)

toc = time.time()
print(time.strftime("%Hh%Mm%Ss", time.gmtime(toc - tic)))