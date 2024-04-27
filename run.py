from nilearn import datasets

urls = ['https://openneuro.org/crn/datasets/ds000244/snapshots/1.0.0/files/MRI_files:anat:sub-01:ses-{:02d}:anat_sub-01_ses-{:02d}_T1w.nii.gz',
        'https://openneuro.org/crn/datasets/ds000244/snapshots/1.0.0/files/MRI_files:func:sub-01:ses-{:02d}:func_sub-01_ses-{:02d}_task-{}_run-{:02d}_bold.nii.gz']

# Fetch the dataset
data = datasets.fetch_openneuro_dataset(urls=urls,dataset_version='ds000244_1.0.0')

# Get the file path of the first fMRI run
fmri_file = data.func[0]

# Load the fMRI data
from nilearn import image
fmri_img = image.load_img(fmri_file)

# Get the shape of the fMRI data
print(fmri_img.shape)