#!/bin/bash

# Define the range of indices for the loop
start_index=0
end_index=9

seeds=(250 100 50 25 42 100 700 300)

# Loop through the list
for seed in "${seeds[@]}"
do
    for ((index=start_index; index<=end_index; index++))
    do
        echo "Running script with index $index"
        python3 train_dsdvdd.py --seed=$seed --data=cifar10 --target=$index
        echo "Finished script with index $index"
    done
done


