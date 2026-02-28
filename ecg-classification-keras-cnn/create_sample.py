import numpy as np

# create random ECG-like 64x64 array
dummy_ecg = np.random.rand(64, 64)

# save as .npy file
np.save("sample_ecg.npy", dummy_ecg)

print("sample_ecg.npy created successfully")