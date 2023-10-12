#!/bin/bash

# Define the command template with a placeholder for the number
command="./TiledGEMM_template -e TiledMatrixMultiplication/Dataset/8/output.raw -i TiledMatrixMultiplication/Dataset/8/input0.raw,TiledMatrixMultiplication/Dataset/8/input1.raw -t vector"

make template

# Run the command and save the output to a file
output_file="./output/output2_16.txt"
$command > "$output_file" 2>&1

# Print a message indicating the command was executed
echo "Executed: $command"
echo "Output saved to: $output_file"
