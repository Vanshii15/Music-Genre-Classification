# Music Genre Classification

A machine learning project to classify music tracks into genres using audio features and deep learning.

## Overview
This project uses a neural network to classify music tracks into different genres based on their audio characteristics. The model is trained on a dataset of music files, extracts relevant features, and predicts the genre of new tracks.

## Features
- Audio feature extraction
- Deep learning model for classification
- Training and evaluation scripts
- Pre-trained model included
- Jupyter notebooks for experimentation

## Project Structure
- `Music_Genre_App.py`: Main application script
- `Train_Music_Genre_Classifier.ipynb`: Notebook for training the model
- `Test_Music_Genre.ipynb`: Notebook for testing the model
- `Trained_model.h5` / `Trained_model.keras`: Pre-trained model files
- `training_hist.json`: Training history
- `requirements.txt`: Python dependencies

## Setup
1. Clone the repository.
2. (Recommended) Create and activate a virtual environment:
   ```powershell
   python -m venv musicenv
   .\musicenv\Scripts\Activate.ps1
   ```
3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

## Usage
- To train the model, run the training notebook:
  - Open `Train_Music_Genre_Classifier.ipynb` in Jupyter and follow the instructions.
- To test or predict genres, use the testing notebook or `Music_Genre_App.py`.

## Requirements
See `requirements.txt` for a list of required Python packages.

## License
This project is for educational purposes.

## Acknowledgements
- Inspired by various open-source music genre classification projects and datasets.