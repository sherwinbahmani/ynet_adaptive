seeds=(1 2 3)
batch_size=4
foldername=/sdd/
out_csv_dir=csv
val_ratio=0.3
datasets=(dataset_filter)
gap_type=(gap)
val_files=(3.25_3.75.pkl)
train_files=(3.25_3.75.pkl)
num_epochs=50
num_train_batches=(1 2 3 4 5 6)
train_nets=(all) #(all encoder modulator)
val_steps=(9 19 29 39 49)
ckpt=/sdd/checkpoints/Seed_1_Train__0.25_0.75__1.25_1.75__2.25_2.75__Val__0.25_0.75__1.25_1.75__2.25_2.75__Val_Ratio_0.3_dataset_ped_biker_gap_weights.pt

for seed in ${seeds[@]}
do
    for dataset in ${datasets[@]}
    do
        for data_type in ${gap_type[@]}
        do
            for train_net in ${train_nets[@]}
            do
                for num in ${num_train_batches[@]}
                do
                    # Few-Shot
                    python train_SDD.py --fine_tune --seed $seed --batch_size $batch_size --foldername $foldername --val_ratio $val_ratio --dataset $dataset --type $data_type --val_files $val_files --out_csv_dir $out_csv_dir --num_epochs $num_epochs --train_files $train_files --num_train_batches $num --train_net $train_net --ckpt $ckpt --val_steps ${val_steps[@]}
                done
            done
        done
    done
done