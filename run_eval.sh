seeds=(1 2 3)
batch_size=4
foldername=/sdd/
out_csv_dir=csv
datasets=(dataset_filter)
gap_type=(gap)
val_files=(3.25_3.75.pkl)
train_nets=all #(all encoder modulator)
ckpt=/sdd/checkpoints/Seed_1_Train__0.25_0.75__1.25_1.75__2.25_2.75__Val__0.25_0.75__1.25_1.75__2.25_2.75__Val_Ratio_0.3_dataset_ped_biker_gap_weights.pt

for seed in ${seeds[@]}
do
    for dataset in ${datasets[@]}
    do
        for data_type in ${gap_type[@]}
        do
            python evaluate_SDD.py --seed $seed --batch_size $batch_size --foldername $foldername --dataset $dataset --type $data_type --val_files $val_files --out_csv_dir $out_csv_dir --ckpt $ckpt
        done
    done
done