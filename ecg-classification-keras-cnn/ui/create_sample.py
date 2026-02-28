import numpy as np

# create smooth low-variation ECG (normal-like)
dummy_ecg = np.ones((64, 64)) * 0.1   # very low signal

np.save("sample_ecg.npy", dummy_ecg)

print("Normal-like ECG sample created")