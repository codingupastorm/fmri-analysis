import os
import numpy as np
import pandas as pd
from nilearn import image, masking
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Set the paths to the Archi spatial and social data
spatial_data_path = r'ds000244-download\sub-01\ses-00\func\sub-01_ses-00_task-ArchiSpatial_acq-ap_bold.nii.gz'
social_data_path = r'ds000244-download\sub-01\ses-00\func\sub-01_ses-00_task-ArchiSocial_acq-ap_bold.nii.gz'

spatial_img = image.load_img(spatial_data_path)
social_img = image.load_img(social_data_path)

print("Transforming Data...")

# Flatten the spatial and social data
spatial_data_flat = spatial_img.get_fdata().reshape(spatial_img.shape[3], -1)
social_data_flat = social_img.get_fdata().reshape(social_img.shape[3], -1)

print("Concating Data...")

# Concatenate the flattened spatial and social data for masking
full_data_flat = np.vstack((spatial_data_flat, social_data_flat))
mask_img = masking.compute_epi_mask(spatial_img)
mask_flat = mask_img.get_fdata().flatten()

# Apply the mask to the flattened data
X = full_data_flat[:, mask_flat.astype(bool)]

# Create labels for each volume (0 for spatial, 1 for social)
y = np.concatenate((np.zeros(spatial_img.shape[3]), np.ones(social_img.shape[3])))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model architecture
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

print("Compiling Model...")

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

print("Training Model...")

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

print("Evaluating Model...")

# Evaluate the model on the testing set
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
