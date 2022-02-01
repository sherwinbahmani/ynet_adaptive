seeds=(1)
batch_size=4
foldername=/sdd/
out_csv_dir=csv
val_ratio=0.3
datasets=(dataset_filter)
gap_type=(gap) #(gap no_gap)
val_files=(0.25_0.75.pkl 1.25_1.75.pkl 2.25_2.75.pkl)
train_files=(0.25_0.75.pkl 1.25_1.75.pkl 2.25_2.75.pkl)
num_epochs=50
train_nets=(all) #(all encoder modulator)

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
                    python train_SDD_trajnet.py --seed $seed --batch_size $batch_size --foldername $foldername --val_ratio $val_ratio --dataset $dataset --type $data_type --val_files $val_files --out_csv_dir $out_csv_dir --num_epochs $num_epochs --train_files $train_files --num_train_batches $num --train_net $train_net --ckpt $ckpt
                done
            done
        done
    done
done
