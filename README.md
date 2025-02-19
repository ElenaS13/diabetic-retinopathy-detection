# Diabetic Retinopathy Detection Project

This repository contains code for preprocessing retinal images for diabetic retinopathy detection.

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/ElenaS13/diabetic-retinopathy-detection
cd diabetic-retinopathy-detection
```

### 2. Set Up Python Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Mac/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 3. Get the Dataset
- Download the dataset from Kaggle: [Diabetic Retinopathy Detection](https://www.kaggle.com/competitions/diabetic-retinopathy-detection/data)
- Place all .jpeg images in the data/raw directory
- Place trainLabels.csv in the data/raw directory

### 4. Project Structure
```
diabetic-retinopathy-detection/
├── data/
│   ├── raw/         # Original images (not in git)
│   └── processed/   # Processed images (not in git)
├── src/
│   ├── __init__.py
│   └── preprocessing.py
├── notebooks/
└── tests/
```

### 5. Running the Pre-processing

To preprocess the images:
```bash
python src/preprocessing.py
```

This will:
- Resize all images to a standard size
- Normalize colors
- Reduce noise
- Save processed images to data/processed/all/
- Processed images are stored in data/processed/all/
- Image labels are in data/raw/trainLabels.csv
- Labels range from 0-4 indicating severity of diabetic retinopathy
