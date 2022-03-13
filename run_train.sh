seeds=(1) # Train the model on different seeds
batch_size=8
num_epochs=50
foldername=sdd_ynet/ # See README.md how to setup this directory
out_csv_dir=csv # /path/to/csv where the output results are written to
val_ratio=0.3 # Split train dataset into a train and val split in case the domains are the same
dataset=dataset_filter/dataset_ped_biker/gap/ # Position the dataset in /path/to/sdd_ynet/
train_files='0.25_0.75.pkl 1.25_1.75.pkl 2.25_2.75.pkl' # Position the dataset files in /path/to/sdd_ynet/{dataset}
val_files='0.25_0.75.pkl 1.25_1.75.pkl 2.25_2.75.pkl' # Position the dataset files in /path/to/sdd_ynet/{dataset}

# train_files='1.25_1.75.pkl' # Position the dataset files in /path/to/sdd_ynet/{dataset}
# val_files='1.25_1.75.pkl' # Position the dataset files in /path/to/sdd_ynet/{dataset}

train_net=all # Train either all parameters, only the encoder or the modulator: (all encoder modulator)
#ckpt=/path/to/ckpt_file # In case you want to resume training, add --ckpt $ckpt


for seed in ${seeds[@]}
do    
    python train_SDD.py --seed $seed --batch_size $batch_size --foldername $foldername --val_ratio $val_ratio --dataset $dataset --val_files $val_files --out_csv_dir $out_csv_dir --num_epochs $num_epochs --train_files $train_files --train_net $train_net
done
