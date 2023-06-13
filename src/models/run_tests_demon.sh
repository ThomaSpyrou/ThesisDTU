#!/bin/bash

# Define the range of indices for the loop
start_index=0
end_index=10
for ((index=start_index; index<=end_index; index++))
do
    echo "Running script with index $index"
    python3 demon.py
    echo "Finished script with index $index"
done