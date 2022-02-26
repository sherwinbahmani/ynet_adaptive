data_raw=/path/to/sdd_ynet/dataset_raw # See README.md on how to download the raw dataset
step=12
window_size=20
stride=20
data_filter=/path/to/sdd_ynet/dataset_filter_custom # Path to new dataset in sdd_ynet directory
labels=(Pedestrian Biker) # Choose a subset from: Biker, Bus, Car, Cart, Pedestrian, Skater
python utils/dataset.py --data_raw $data_raw --step $step --window_size $window_size --stride $stride --data_filter $data_filter --labels $labels
