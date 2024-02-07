import numpy as np
noise_level = 0.5
random_noise = np.random.uniform(-noise_level, noise_level, size=(290, 3)) 
outliers = np.random.uniform(-noise_level, noise_level, size=(10, 3)) 

toge = np.vstack((random_noise,outliers))

te = np.ones((300,3))
print(toge.shape)
print(te.shape)
print((te+toge).shape)