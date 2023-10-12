#!/bin/bash

# Define the command template with a placeholder for the number
command="./VectorAdd_template -e VectorAdd/Dataset/NUM/output.raw -i VectorAdd/Dataset/NUM/input0.raw,VectorAdd/Dataset/NUM/input1.raw -t vector"

# Loop from 0 to 9 to replace the placeholder and run the command
for number in {0..9}; do
    # Replace the placeholder with the current number
    modified_command="${command//NUM/$number}"

    # Run the command and save the output to a file
    output_file="output_$number.txt"
    $modified_command > "$output_file" 2>&1

    # Print a message indicating the command was executed
    echo "Executed: $modified_command"
    echo "Output saved to: $output_file"
done