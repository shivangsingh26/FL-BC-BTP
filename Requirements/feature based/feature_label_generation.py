import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths to the finger vein dataset directory
data_dir = "FVData"
train_dir = "FVData/train"
test_dir = "FVData/test"

# Image preprocessing function
def preprocess_image(img):
    # Customize this function based on your preprocessing requirements
    return img / 255.0  # Normalize to the range [0, 1]

# from tensorflow.keras.preprocessing.image import ImageDataGenerator

batch_size = 32
image_size = (224, 224)

# Use ImageDataGenerator to generate labels from the directory structure
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Save train labels to train_labels.npy
train_labels = np.array(train_generator.classes)
np.save("train_labels.npy", train_labels)

# Save test labels to test_labels.npy
test_labels = np.array(test_generator.classes)
np.save("test_labels.npy", test_labels)

# Print the class indices
print("Class Indices:", train_generator.class_indices)

unique_train_labels = np.unique(train_labels)
unique_test_labels = np.unique(test_labels)

print("Unique Labels in Training Set:", unique_train_labels)
print("Unique Labels in Testing Set:", unique_test_labels)
