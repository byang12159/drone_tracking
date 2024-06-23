import csv
import matplotlib.pyplot as plt
import numpy as np

def read_data_from_csv(csv_file):
    data = []

    with open(csv_file, 'r', newline='') as csvfile:
        csvreader = csv.reader(csvfile)

        current_run = None
        for row in csvreader:
            if row and row[0].startswith("Run #"):
                # If we encounter a new "Run #" header, create a new list for its data
                current_run = []
                data.append(current_run)
            elif current_run is not None:
                # If current_run is set (meaning we are collecting data for a run), append the row
                values = [float(val) for val in row]
                current_run.append(values)

    return data

# Example usage
csv_file = 'data.csv'
data = read_data_from_csv(csv_file)

timestep = 0.01

# Display the extracted data
fig, (varx, vary, varz) = plt.subplots(3, 1, figsize=(14, 10))
for idx, run_data in enumerate(data):
    data_unit =np.array(data[idx])
    
    times = np.arange(0,data_unit.shape[0])*timestep
    varx.plot(times, data_unit[:,0], label = "Var x")
    varx.legend()
    vary.plot(times, data_unit[:,1], label = "Var y")    
    vary.legend()
    varz.plot(times, data_unit[:,2], label = "Var z")
    varz.legend()
plt.show()