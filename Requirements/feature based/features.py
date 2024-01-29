import numpy as np

# Load features from the .npy files
train_features = np.load("train_features.npy")
test_features = np.load("test_features.npy")

# Print the shape of the loaded features
print("Train Features Shape:", train_features.shape)
print("Test Features Shape:", test_features.shape)

# Display the first few rows of the features
print("Train Features:")
print(train_features)

print("Test Features:")
print(test_features)

#%%
