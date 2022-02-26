seeds=(1)
batch_size=4
foldername=/path/to/sdd_ynet/ # See README.md how to setup this directory
out_csv_dir=csv # /path/to/csv where the output results are written to
dataset=dataset_filter/gap # Position the dataset in /path/to/sdd_ynet/
val_files=(3.25_3.75.pkl) # Position the dataset files in /path/to/sdd_ynet/{dataset}
train_net=all # Train either all parameters, only the encoder or the modulator: (all encoder modulator)
ckpt=/path/to/checkpoint.pt # For example: /path/to/Seed_1_Train__0.25_0.75__1.25_1.75__2.25_2.75__Val__0.25_0.75__1.25_1.75__2.25_2.75__Val_Ratio_0.3_dataset_ped_biker_gap_weights.pt

for seed in ${seeds[@]}
do
    python evaluate_SDD.py --seed $seed --batch_size $batch_size --foldername $foldername --dataset $dataset --val_files $val_files --out_csv_dir $out_csv_dir --ckpt $ckpt
done