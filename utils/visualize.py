import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import seaborn as sns
import os
import pathlib
import argparse

labels = {"all": "Vanilla fine-tuning", "modulator": "Modular adaptation"}

def create_few_shot_plot(results_dir, out_dir, fontsize=16):
    update_modes = sorted(os.listdir(results_dir))
    ades = {}
    for update_mode in update_modes:
        update_mode_dir = os.path.join(results_dir, update_mode)
        seeds = os.listdir(update_mode_dir)
        ades[update_mode] = {}
        for seed in seeds:
            seed_dir = os.path.join(update_mode_dir, seed)
            num_files = os.listdir(seed_dir)
            for num_file in num_files:
                num = int(num_file.split('.csv')[0])
                num_path = os.path.join(seed_dir, num_file)
                ade = float(pd.read_csv(num_path).values[0][0]) #float(pd.read_csv(num_path).columns[0]) 
                if num not in ades[update_mode]:
                    ades[update_mode][num] = []
                ades[update_mode][num].append(ade)
            zero_shot_path = results_dir.split("/")
            zero_shot_path[-2] = "None"
            zero_shot_path += ['eval', seed, '0.csv']
            zero_shot_path = '/'.join(zero_shot_path)
            if os.path.isfile(zero_shot_path):
                ade = float(pd.read_csv(zero_shot_path).values[0][0]) #float(pd.read_csv(num_path).columns[0]) 
                num = 0
                if num not in ades[update_mode]:
                    ades[update_mode][num] = []
                ades[update_mode][num].append(ade)

    f, ax = plt.subplots(figsize=(4.96, 2.77))
    for train_name, train_vals in ades.items():
        v = [i for j in list(train_vals.values()) for i in j]
        k =[j for j in list(train_vals.keys()) for _ in range(len(list(train_vals.values())[0]))]
        df = pd.DataFrame({'x': k, 'y': v})
        sns.lineplot(data=df, x='x', y='y', label=labels[train_name], ax=ax, marker="o")
        sns.despine()
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    plt.ylabel('ADE', fontsize=fontsize)
    plt.xlabel('# Batches', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    plt.savefig(f'{out_dir}/result.pdf', bbox_inches='tight', pad_inches=0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default='csv/dataset_filter/dataset_ped_biker/gap/3.25_3.75/3.25_3.75', type=str)
    parser.add_argument("--out_dir", default='plots', type=str)
    args = parser.parse_args()
    create_few_shot_plot(args.results_dir, args.out_dir)