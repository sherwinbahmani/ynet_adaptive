import pandas as pd
import yaml
from model import YNet
import time
import os
from utils.dataset import set_random_seeds
from utils.parser import val_parser
from utils.write_files import write_csv

tic = time.time()
args = val_parser()
set_random_seeds(args.seed)
CONFIG_FILE_PATH = os.path.join('config', 'sdd_raw_eval.yaml')  # yaml config file containing all the hyperparameters
DATASET_NAME = 'sdd'


with open(CONFIG_FILE_PATH) as file:
    params = yaml.load(file, Loader=yaml.FullLoader)

if args.gpu != None:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

params['segmentation_model_fp'] = os.path.join(args.foldername, 'ynet_additional_files', 'segmentation_models', 'SDD_segmentation.pth')

TEST_IMAGE_PATH = os.path.join(args.foldername, 'dataset_raw', 'annotations')
assert os.path.isdir(TEST_IMAGE_PATH), 'raw data dir error'
DATA_PATH = os.path.join(args.foldername, args.dataset)

df_test = pd.concat([pd.read_pickle(os.path.join(DATA_PATH, test_file)) for test_file in args.val_files])

model = YNet(obs_len=params['OBS_LEN'], pred_len=params['PRED_LEN'], params=params)
if args.ckpt is not None:
    if args.train_net == "modulator":
	    model.model.initialize_style()
    model.load(args.ckpt)
    print(f"Loaded checkpoint {args.ckpt}")
else:
    raise ValueError("No checkpoint given!")

ade, fde = model.evaluate(df_test, params, image_path=TEST_IMAGE_PATH,
            batch_size=args.batch_size, rounds=args.rounds, 
            num_goals=params['NUM_GOALS'], num_traj=params['NUM_TRAJ'], device=None, dataset_name=DATASET_NAME,
            use_raw_data=params['use_raw_data'], with_style=args.train_net == "modulator")


toc = time.time()
print(time.strftime("%Hh%Mm%Ss", time.gmtime(toc - tic)))

if args.out_csv_dir is not None:
    write_csv(args.out_csv_dir, args.seed, ade, fde, 0, 0, "eval", args.dataset,
              args.val_files, None)