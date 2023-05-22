#!/bin/bash

# Define the range of indices for the loop
start_index=0
end_index=9

# Loop through the indices
for ((index=start_index; index<=end_index; index++))
do
    echo "Running script with index $index"
    python3 train_dsdvdd.py --seed=1000 --data=cifar10 --target=$index
    echo "Finished script with index $index"
done
