import numpy as np
import pickle
import matplotlib.pyplot as plt

# File path from which to load the data
file_path = 'data.pickle'

# Open the file in binary read mode
with open(file_path, 'rb') as file:
    # Load the data from the file
    data = pickle.load(file)

# Print the loaded data
print("Loaded data:", data)

print("shape ",len(data))

accelerations=[]
lookaheads = []
for i in range(0, len(data), 3):
    print(data[i])
    
    print("Mean lookahead steps: ",np.mean(data[i+1]))

    accelerations.append(data[i])
    lookaheads.append(np.mean(data[i+1]))

print(accelerations)
print(lookaheads)

plt.plot(accelerations,lookaheads)
# Add labels and title
plt.xlabel('Acceleration Bounds')
plt.ylabel('Number of Time steps Predicted')
plt.title('#Time-steps Predicted within 0.5m radius Sphere')

# Display the plot
plt.show()