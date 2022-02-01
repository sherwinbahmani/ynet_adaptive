from typing import List
import pathlib
import os
import csv

def write_csv(out_dir: str, seed: int, ade: float, fde: float, num_epochs: int, num_batches: int, train_net: str, dataset: str,
              val_files: List[str], train_files: List[str] = None, train_ade: float = None, train_fde: float = None):
    val_name = f"{'_'.join(['_'+f.split('.pkl')[0]+'_' for f in val_files])}"
    if train_files is not None:
        train_name = f"{'_'.join(['_'+f.split('.pkl')[0]+'_' for f in train_files])}"
    else:
        train_name = "None"
    out_name = f"{num_batches}.csv"
    out_dir = os.path.join(out_dir, dataset, train_name.strip("_"), val_name.strip("_"), train_net, str(seed))
    out_path = os.path.join(out_dir, out_name)
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_data = [round_val(ade), round_val(fde), num_epochs, num_batches, train_net, dataset, val_name, train_name]
    if train_ade is not None:
        out_data.append(round_val(train_ade))
    if train_fde is not None:
        out_data.append(round_val(train_fde))
    out_data = convert_to_str(out_data)
    with open(out_path, 'w') as f:
        wr = csv.writer(f, dialect='excel')
        wr.writerows(out_data)
    pass

def round_val(num: float, ndig: int = 4):
    if num is None:
        return 0.0
    else:
        return str(round(num, ndig))

def convert_to_str(in_list: List):
    out_list = []
    for i in in_list:
        if not isinstance(i, str):
            i = str(i)
        out_list.append([i])
    return out_list
