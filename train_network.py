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

spatial_imgs = image.load_img(spatial_data_path)
social_imgs = image.load_img(social_data_path)

# Concatenate the volumes into a single 4D array for each condition
spatial_data = image.concat_imgs(spatial_imgs)
social_data = image.concat_imgs(social_imgs)

# Apply a mask to extract relevant brain regions (e.g., using a pre-defined mask or creating one)
mask_img = masking.compute_epi_mask(spatial_data)
spatial_data_masked = masking.apply_mask(spatial_data, mask_img)
social_data_masked = masking.apply_mask(social_data, mask_img)

# Prepare the labels for each condition
spatial_labels = np.zeros(len(spatial_files))
social_labels = np.ones(len(social_files))

# Combine the data and labels
X = np.vstack((spatial_data_masked, social_data_masked))
y = np.concatenate((spatial_labels, social_labels))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape the data to include a single channel dimension
X_train = X_train.reshape((-1, *spatial_data.shape[1:], 1))
X_test = X_test.reshape((-1, *spatial_data.shape[1:], 1))

# Define the CNN model architecture
model = Sequential()
model.add(Conv3D(32, (3, 3, 3), activation="relu", input_shape=X_train.shape[1:]))
model.add(MaxPooling3D((2, 2, 2)))
model.add(Conv3D(64, (3, 3, 3), activation="relu"))
model.add(MaxPooling3D((2, 2, 2)))
model.add(Conv3D(128, (3, 3, 3), activation="relu"))
model.add(MaxPooling3D((2, 2, 2)))
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(X_train, y_train, batch_size=32, epochs=10, validation_split=0.1)

# Evaluate the model on the testing set
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)