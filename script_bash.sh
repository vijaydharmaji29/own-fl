#!/bin/bash

# Number of terminals you want to open
num_terminals=$1
start_port=50000

# Array to store background process IDs
pids=()

# Loop to run Python scripts concurrently
for ((i=1; i<=$num_terminals; i++))
do
    result_port=$((start_port + i))
    echo "Running test${i}.py"
    python -m image_classification.classification_engine 1 ${result_port} a${i} 20 0 2>&1 | tee output_${i}.log &   # Execute in background, tee to print and log
    pids+=($!)  # Store process ID of each background task
done

# Wait for all background processes to finish
for pid in "${pids[@]}"
do
    wait "$pid"
done

echo "All tests have finished."
