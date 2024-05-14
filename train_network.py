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

full_data = image.concat_imgs([spatial_img, social_img])

mask_img = masking.compute_epi_mask(full_data)
spatial_data_masked = masking.apply_mask(spatial_img, mask_img)
social_data_masked = masking.apply_mask(social_img, mask_img)

# Create labels for each volume (0 for spatial, 1 for social)
spatial_labels = np.zeros(spatial_data_masked.shape[0])
social_labels = np.ones(social_data_masked.shape[0])

# Combine the masked data and labels
X = np.vstack((spatial_data_masked, social_data_masked))
y = np.concatenate((spatial_labels, social_labels))

print("Splitting Data...")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Reshaping Data...")

# Reshape the data to add a single channel dimension
X_train = X_train.reshape((-1, *mask_img.shape))
X_test = X_test.reshape((-1, *mask_img.shape))

# Define the CNN model architecture
model = Sequential()
model.add(Conv3D(32, (3, 3, 3), activation='relu', input_shape=X_train.shape[1:]))
model.add(MaxPooling3D((2, 2, 2)))
model.add(Conv3D(64, (3, 3, 3), activation='relu'))
model.add(MaxPooling3D((2, 2, 2)))
model.add(Conv3D(128, (3, 3, 3), activation='relu'))
model.add(MaxPooling3D((2, 2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

print("Compiling Model...")

# Compile the model
model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

print("Training Model...")

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

print("Evaluating Model...")

# Evaluate the model on the testing set
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")