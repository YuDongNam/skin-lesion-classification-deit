# Data Directory

This directory contains the datasets used for skin lesion classification.

## Dataset Information

### HAM10000 Dataset
- **Source**: [ISIC Archive](https://www.isic-archive.com/#!/topWithHeader/onlyHeaderTop/gallery)
- **Description**: 10,015 dermatoscopic images of common pigmented skin lesions
- **Classes**: 7 classes of skin lesions
- **Format**: JPG images with metadata CSV

### ISIC 2019 Dataset
- **Source**: [ISIC 2019 Challenge](https://challenge.isic-archive.com/landing/2019/)
- **Description**: Additional skin lesion images for training and testing
- **Classes**: Same 7 classes as HAM10000
- **Format**: JPG images with metadata CSV

## Class Labels

| ID | Abbreviation | Full Name | Description |
|----|-------------|-----------|-------------|
| 0 | nv | Melanocytic nevus | Benign mole (모반) |
| 1 | mel | Melanoma | Malignant melanoma (흑색종) |
| 2 | bkl | Benign keratosis-like lesions | Benign keratosis (양성 각화증) |
| 3 | bcc | Basal cell carcinoma | Basal cell carcinoma (기저세포암) |
| 4 | akiec | Actinic keratosis | Actinic keratosis (광선각화증) |
| 5 | vasc | Vascular lesions | Vascular lesions (혈관병변) |
| 6 | df | Dermatofibroma | Dermatofibroma (섬유종) |

## Data Structure

```
data/
├── HAM10000_metadata.csv          # HAM10000 dataset metadata
├── train.csv                      # ISIC 2019 training metadata
├── images/                        # Image files (not included in repo)
│   ├── ISIC_0027419.jpg
│   ├── ISIC_0025030.jpg
│   └── ...
└── README.md                      # This file
```

## Data Access

Due to the large size of the image files, they are not included in this repository. To obtain the datasets:

1. **HAM10000 Dataset**:
   - Visit the [ISIC Archive](https://www.isic-archive.com/#!/topWithHeader/onlyHeaderTop/gallery)
   - Download the HAM10000 dataset
   - Extract images to `data/images/` directory

2. **ISIC 2019 Dataset**:
   - Visit the [ISIC 2019 Challenge](https://challenge.isic-archive.com/landing/2019/)
   - Download the training and test datasets
   - Extract images to `data/images/` directory

## Data Preprocessing

The datasets undergo the following preprocessing steps:

1. **Hair Removal**: Morphological transformations to remove hair artifacts
2. **Resizing**: Images are resized to 384x384 pixels for DeiT model
3. **Normalization**: Pixel values are normalized to [0, 1] range
4. **Data Augmentation**: Applied during training (rotation, flipping, etc.)

## Usage

The data preprocessing functions in `src/data_preprocessing.py` handle:
- Loading metadata from CSV files
- Creating image paths
- Applying hair removal preprocessing
- Creating train/test splits
- Generating class distribution statistics

## Notes

- The HAM10000 dataset is imbalanced with significantly more nevus (nv) samples
- Hair removal preprocessing was found to hurt performance and is optional
- Images are stored in BGR format for OpenCV compatibility
- All images are converted to RGB format for model input
