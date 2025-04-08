from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import io
import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import base64
import torch
from torchvision import transforms
import torch.nn.functional as F

app = Flask(__name__)

# Load Pneumonia Detection Model
try:
    pneumonia_model = load_model("pneumonia_module/pneumonia_model.h5")
    print("Pneumonia model loaded successfully.")
except Exception as e:
    print(f"Error loading pneumonia model: {e}")
    pneumonia_model = None

# Load Pneumonia Treatment Recommendation Model and Label Encoders
try:
    model_and_encoders = joblib.load('pneumonia_module/Pnemonia_recommendation_model_and_encoders.pkl')
    recommendation_model = model_and_encoders['model']
    label_encoders = model_and_encoders['label_encoders']
    print("Recommendation model and label encoders loaded successfully.")
except Exception as e:
    print(f"Error loading recommendation model or label encoders: {e}")
    recommendation_model = None
    label_encoders = None

# Load Tumor Detection Model
class TumorClassifier(torch.nn.Module):
    def __init__(self, num_classes):
        super(TumorClassifier, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(32 * 56 * 56, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

try:
    tumor_model = TumorClassifier(num_classes=4)
    tumor_model.load_state_dict(torch.load('tumor_module/brain_tumor_classifier.pth', map_location=torch.device('cpu')))
    tumor_model.eval()
    print("Tumor model loaded successfully.")
except Exception as e:
    print(f"Error loading tumor model: {e}")
    tumor_model = None

# Preprocess image for Pneumonia Detection
def preprocess_pneumonia_image(img_data):
    try:
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error preprocessing pneumonia image: {e}")
        return None

# Preprocess image for Tumor Detection
def preprocess_tumor_image(img_data):
    try:
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        img = img.resize((224, 224))
        img_tensor = transforms.ToTensor()(img).unsqueeze(0)
        return img_tensor
    except Exception as e:
        print(f"Error preprocessing tumor image: {e}")
        return None

# Encode image as base64
def encode_image_to_base64(img_data):
    try:
        img = Image.open(io.BytesIO(img_data))
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image to base64: {e}")
        return None

# Homepage
@app.route('/')
def home():
    return render_template('index.html')

# Pneumonia Detection Page
@app.route('/pneumonia', methods=['GET', 'POST'])
def pneumonia():
    pneumonia_result = None
    confidence = None
    show_recommendation_button = False
    show_recommendation_form = False
    uploaded_image_base64 = None
    recommended_treatment = None
    recommended_pills = None
    recommended_days = None
    error = None

    if request.method == 'POST':
        # Pneumonia Detection
        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                try:
                    img_data = file.read()
                    uploaded_image_base64 = encode_image_to_base64(img_data)

                    if uploaded_image_base64 is None:
                        error = "Error processing image."
                    else:
                        img_array = preprocess_pneumonia_image(img_data)
                        if img_array is None:
                            error = "Error preprocessing image."
                        else:
                            if pneumonia_model is None:
                                error = "Pneumonia model not loaded."
                            else:
                                prediction = pneumonia_model.predict(img_array)[0][0]
                                confidence = round(float(prediction) * 100, 2) if prediction > 0.5 else round((1 - float(prediction)) * 100, 2)
                                pneumonia_result = "🚨 Pneumonia Detected!" if prediction > 0.5 else "✅ Normal X-ray"
                                
                                # Show recommendation button only if pneumonia is detected
                                if prediction > 0.5:
                                    show_recommendation_button = True
                except Exception as e:
                    print(f"Error during pneumonia detection: {e}")
                    error = "An error occurred during pneumonia detection."

        # Trigger Recommendation Form
        if 'get_recommendation' in request.form:
            uploaded_image_base64 = request.form.get('uploaded_image_base64', '')
            if uploaded_image_base64:
                show_recommendation_form = True

        # Treatment Recommendation
        if 'age' in request.form:
            try:
                age = int(request.form['age'])
                gender = request.form['gender']
                pregnancy_status = request.form['pregnancy_status']
                smoking_history = request.form['smoking_history']
                previous_antibiotic_use = request.form['previous_antibiotic_use']
                hospitalization_status = request.form['hospitalization_status']
                uploaded_image_base64 = request.form.get('uploaded_image_base64', '')

                # Encode fields using label encoders
                gender_encoded = label_encoders['Gender'].transform([gender])[0]
                pregnancy_status_encoded = label_encoders['Pregnancy Status'].transform([pregnancy_status])[0]
                smoking_history_encoded = label_encoders['Smoking History'].transform([smoking_history])[0]
                previous_antibiotic_use_encoded = label_encoders['Previous Antibiotic Use'].transform([previous_antibiotic_use])[0]
                hospitalization_status_encoded = label_encoders['Hospitalization Status'].transform([hospitalization_status])[0]

                # Create input dataframe
                input_data = pd.DataFrame({
                    'Age': [age],
                    'Gender': [gender_encoded],
                    'Pregnancy Status': [pregnancy_status_encoded],
                    'Smoking History': [smoking_history_encoded],
                    'Previous Antibiotic Use': [previous_antibiotic_use_encoded],
                    'Hospitalization Status': [hospitalization_status_encoded]
                })

                # Predict recommendation
                if recommendation_model is None:
                    error = "Recommendation model not loaded."
                else:
                    prediction = recommendation_model.predict(input_data)
                    recommended_treatment_encoded = int(np.round(prediction[0][0]))
                    recommended_pills = int(np.round(prediction[0][1]))
                    recommended_days = int(np.round(prediction[0][2]))

                    recommended_treatment = label_encoders['Recommended Treatment'].inverse_transform([recommended_treatment_encoded])[0]
                    show_recommendation_form = True
            except Exception as e:
                print(f"Error during treatment recommendation: {e}")
                error = "An error occurred during treatment recommendation."

    return render_template('pneumonia.html',
                           pneumonia_result=pneumonia_result,
                           confidence=confidence,
                           show_recommendation_button=show_recommendation_button,
                           show_recommendation_form=show_recommendation_form,
                           uploaded_image_base64=uploaded_image_base64,
                           recommended_treatment=recommended_treatment,
                           recommended_pills=recommended_pills,
                           recommended_days=recommended_days,
                           error=error)

# Tumor Detection Page
@app.route('/tumor', methods=['GET', 'POST'])
def tumor():
    prediction = None
    confidence = None
    image_data = None
    error = None

    if request.method == 'POST':
        if 'file' not in request.files:
            error = "No file uploaded"
            return render_template('tumor.html', error=error)
        
        file = request.files['file']
        if file.filename == '':
            error = "No file selected"
            return render_template('tumor.html', error=error)

        try:
            # Read and encode the image
            img_data = file.read()
            img = Image.open(io.BytesIO(img_data)).convert('RGB')
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            image_data = base64.b64encode(buffered.getvalue()).decode('utf-8')

            # Preprocess for model (match your original app)
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            img_tensor = transform(img).unsqueeze(0)

            if tumor_model is None:
                error = "Tumor model not loaded"
                return render_template('tumor.html', error=error)

            # Make prediction
            with torch.no_grad():
                output = tumor_model(img_tensor)
                probabilities = F.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                prediction = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary'][predicted.item()]
                confidence = confidence.item() * 100

        except Exception as e:
            error = f"Error processing image: {str(e)}"

    return render_template('tumor.html',
                        prediction=prediction,
                        confidence=confidence,
                        image_data=image_data,
                        error=error)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))

