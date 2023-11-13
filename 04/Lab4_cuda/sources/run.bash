#!/bin/bash

# Define the command template with a placeholder for the number
command="./Convolution_template -e Convolution/Dataset/NUM/output.ppm -i Convolution/Dataset/NUM/input0.ppm,Convolution/Dataset/NUM/input1.raw -o output/NUM.ppm -t image"

make template
rm -r output
mkdir output

# Loop from 0 to 9 to replace the placeholder and run the command
for number in {0..6}; do
    # Replace the placeholder with the current number
    modified_command="${command//NUM/$number}"

    # Run the command and save the output to a file
    output_file="./output/output_$number.txt"
    $modified_command > "$output_file" 2>&1

    # Print a message indicating the command was executed
    echo "Executed: $modified_command"
    echo "Output saved to: $output_file"
done
