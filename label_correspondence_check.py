import numpy as np

# Load features and labels
train_features = np.load("train_features.npy")
test_features = np.load("test_features.npy")
train_labels = np.load("train_labels.npy")
test_labels = np.load("test_labels.npy")

# Verify the lengths match
assert len(train_features) == len(train_labels)
assert len(test_features) == len(test_labels)

# Check a few samples
for i in range(5):  # Adjust the range based on your needs
    print(f"Sample {i + 1} - Train: {train_features[i]} - Label: {train_labels[i]}")
    print(f"Sample {i + 1} - Test: {test_features[i]} - Label: {test_labels[i]}")
