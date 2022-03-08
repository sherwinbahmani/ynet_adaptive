seeds=(1) # Fine-tune the model on different seeds
batch_size=4
num_epochs=50
foldername=sdd_ynet/ # See README.md how to setup this directory
out_csv_dir=csv # /path/to/csv where the output results are written to
val_ratio=0.3 # Split train dataset into a train and val split in case the domains are the same
dataset=dataset_filter/dataset_ped_biker/gap/ # Position the dataset in /path/to/sdd_ynet/
val_files='3.25_3.75.pkl' # Position the dataset files in /path/to/sdd_ynet/{dataset}
train_files='3.25_3.75.pkl' # Position the dataset files in /path/to/sdd_ynet/{dataset}
num_train_batches=(1 2 3 4 5 6) # Fine-tune the model with a given number of batches
train_net=all # Train either all parameters, only the encoder or the modulator: (all encoder modulator)
val_steps=(9 19 29 39 49) # Evaluate the model on given steps during training
ckpt=/path/to/checkpoint.pt # For example: /path/to/Seed_1_Train__0.25_0.75__1.25_1.75__2.25_2.75__Val__0.25_0.75__1.25_1.75__2.25_2.75__Val_Ratio_0.3_dataset_ped_biker_gap_weights.pt

for seed in ${seeds[@]}
do
    for num in ${num_train_batches[@]}
    do
        python train_SDD.py --fine_tune --seed $seed --batch_size $batch_size --foldername $foldername --val_ratio $val_ratio --dataset $dataset --val_files $val_files --out_csv_dir $out_csv_dir --num_epochs $num_epochs --train_files $train_files --num_train_batches $num --train_net $train_net --ckpt $ckpt --val_steps ${val_steps[@]}
    done
done