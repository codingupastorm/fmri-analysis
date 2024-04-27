import os
from nilearn import image

# Specify the local path to the dataset
dataset_path = 'ds000244-download'

# Specify the relative path to the first fMRI run
fmri_file = r'sub-01\ses-00\func\sub-01_ses-00_task-ArchiSocial_acq-ap_bold.nii.gz'

# Construct the full path to the fMRI file
fmri_path = os.path.join(dataset_path, fmri_file)

# Load the fMRI data
fmri_img = image.load_img(fmri_path)

# Get the shape of the fMRI data
print(f"fMRI data shape: {fmri_img.shape}")
print(f"fMRI data shape: {fmri_img.header.get_zooms()[-1]}")