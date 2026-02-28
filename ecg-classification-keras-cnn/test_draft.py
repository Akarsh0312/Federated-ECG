from drift import detect_drift

print("Test case 1 (confidence = 0.52)")
print(detect_drift(0.52))

print("\nTest case 2 (confidence = 0.97)")
print(detect_drift(0.97))