import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--foldername", default='/sdd/', type=str)
    parser.add_argument("--save_every", default=50, type=int, help="save checkpoint every n epochs on top of best one")
    parser.add_argument("--gpu", default=None, type=int, help='gpu id to use')
    # data
    parser.add_argument("--val_ratio", default=0.3, type=float)
    parser.add_argument("--dataset", default='dataset_ped_biker', type=str)
    parser.add_argument("--type", default='gap', type=str, choices=["gap", "no_gap"], help='gap or no_gap between velocity ranges')
    parser.add_argument("--val_files", default=["0.25_0.75.pkl", "1.25_1.75.pkl", "2.25_2.75.pkl"], type=str, nargs="+")
    parser.add_argument("--out_csv_dir", default="csv", type=str, help="If not None, will write a csv with results in this dir")
    # model
    parser.add_argument("--train_net", default="all", choices=["all", "encoder", "modulator"], type=str, help="train all parameters or only encoder or with modulator")
    parser.add_argument("--ckpt", default=None, type=str, help='path to checkpoint')
    parser.add_argument("--rounds", default=1, type=int, help='number of rounds in stochastics eval process')
    return parser

def train_parser():
    parser = get_parser()
    parser.add_argument("--num_epochs", default=1, type=int)
    parser.add_argument("--train_files", default=["0.25_0.75.pkl", "1.25_1.75.pkl", "2.25_2.75.pkl"], type=str, nargs="+")
    parser.add_argument("--num_train_batches", default=1, type=int, help="limited number of batches for each training agent, None means no limit")
    parser.add_argument("--fine_tune", action="store_true")
    parser.add_argument("--val_steps", default=None, type=int, nargs="+", help="Evaluate during training every n epochs, for None every epoch")
    parser.add_argument("--learning_rate", default=0.01, type=float)
    return parser.parse_args()

def val_parser():
    parser = get_parser()
    return parser.parse_args()