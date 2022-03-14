seeds=(1 2 3)
batch_size=8
foldername=sdd_ynet/ # See README.md how to setup this directory
out_csv_dir=csv # /path/to/csv where the output results are written to
val_ratio=0.3 # Only use a subset of the dataset for evaluation to make it comparable to fine-tuning
dataset=dataset_filter/dataset_ped_biker/gap/ # Position the dataset in /path/to/sdd_ynet/
val_files='3.25_3.75.pkl' # Position the dataset files in /path/to/sdd_ynet/{dataset}
ckpt=ckpts/Seed_1_Train__0.25_0.75__1.25_1.75__2.25_2.75__Val__0.25_0.75__1.25_1.75__2.25_2.75__Val_Ratio_0.3_dataset_ped_biker_gap_weights.pt # Pre-trained model

for seed in ${seeds[@]}
do
    python evaluate_SDD.py --seed $seed --batch_size $batch_size --foldername $foldername --dataset $dataset --val_files $val_files --out_csv_dir $out_csv_dir --ckpt $ckpt
done