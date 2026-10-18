# Floraexa – Plant Identifier & Disease Detector

## About the Project
Floraexa is a plant disease detection web application that uses a Convolutional Neural Network (CNN) model to analyze uploaded leaf images and predict possible plant diseases.

## Features
- Upload a plant leaf image.
- Predict the possible plant disease using a trained deep learning model.
- Display the predicted class and confidence score.
- Simple and user-friendly web interface.

## Technologies Used
- Python
- FastAPI
- TensorFlow / Keras
- CNN (EfficientNet model)
- HTML, CSS, JavaScript
- Pillow and NumPy

## Project Structure
- `app.py` – Backend and prediction logic
- `models/` – Trained model and class labels
- `templates/` – HTML templates
- `static/` – CSS and JavaScript files

## How to Run
1. Install the required Python packages.
2. Open the project folder in the terminal.
3. Run:

   `python -m uvicorn app:app --reload`

4. Open the local URL shown in the terminal.

## Note
Predictions are model-generated estimates and may not always be correct. The project is intended for educational purposes.

Floraexa is an AI-powered web application that identifies plant species and detects common leaf diseases using a Convolutional Neural Network (CNN). It uses FastAPI to process uploaded leaf images and display predictions, helping users understand plant health and recognise potential diseases early.
