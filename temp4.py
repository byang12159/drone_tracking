# Define the range and increment
import numpy as np 
start = -9
stop = 9
increment = 1.0

# Initialize an empty list to store the combinations
combinations = []

# Iterate through x, y, and z dimensions
for x in [i * increment for i in range(int((stop - start) / increment) + 1)]:
    for y in [i * increment for i in range(int((stop - start) / increment) + 1)]:
        for z in [i * increment for i in range(int((stop - start) / increment) + 1)]:
            # Calculate the actual value of x, y, and z
            x_value = start + x * increment
            y_value = start + y * increment
            z_value = start + z * increment
            # Append the combination to the list
            combinations.append((x_value, y_value, z_value))

# Print the combinations
combinations=np.array(combinations)
print(combinations.shape)
