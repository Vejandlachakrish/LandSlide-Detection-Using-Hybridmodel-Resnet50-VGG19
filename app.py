from flask import Flask, render_template, request, redirect, url_for
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import os
import uuid
import logging
import torch.nn as nn
from torchvision import models
from threading import Lock

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize a lock for thread safety
model_lock = Lock()


# Define your model architecture with ResNet50 and EfficientNet
class DanceClassificationModel(nn.Module):
    def __init__(self, num_classes=8):
        super(DanceClassificationModel, self).__init__()
        self.resnet50 = models.resnet50(weights='DEFAULT')
        self.efficientnet = models.efficientnet_b0(weights='DEFAULT')

        # Replace the classifiers to match number of classes
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, num_classes)
        self.efficientnet.classifier[-1] = nn.Linear(
            self.efficientnet.classifier[-1].in_features, num_classes)

    def forward(self, x):
        resnet_features = self.resnet50(x)
        efficientnet_features = self.efficientnet(x)
        combined_features = (resnet_features + efficientnet_features) / 2 
        return combined_features


model = DanceClassificationModel(num_classes=8)

# Load model weights
model_path = './land-slide-1.pth'

model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
model.eval()  # Set the model to evaluation mode

# Define image preprocessing with data augmentation for inference
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to fit model input
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

# Define class names
class_names = [
    'LandSlide',
    'Non-LandSlide'
]


def model_prediction(image):
    image = preprocess(image)  
    input_tensor = image.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        predictions = model(input_tensor)
    result_index = torch.argmax(predictions, dim=1).item() 
    return class_names[result_index]  # Return class name directly


logging.basicConfig(level=logging.DEBUG)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/handle_contact', methods=['POST'])
def handle_contact():
    name = request.form['name']
    email = request.form['email']
    message = request.form['message']
    app.logger.info(f'Received contact form submission: Name={name}, \
    Email={email}, Message={message}')
    return redirect(url_for('contact'))


@app.route('/back', methods=['GET'])
def back():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    model.eval()
    if 'image' not in request.files:
        return redirect(request.url)
    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)
    if file:
        # Save the image to the upload folder
        filename = f"{uuid.uuid4().hex}.png"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            # Open the image from the uploaded file
            image = Image.open(io.BytesIO(file.read())).convert('RGB')
            app.logger.debug(f'Input image size: {image.size}')
            # Save the uploaded image for reference (if needed)
            image.save(file_path)

            # Make a prediction
            prediction = model_prediction(image)
            app.logger.debug(
                f'Image prediction: {prediction} for image: {filename}')
            return render_template(
                'result.html', prediction=prediction, image_url=url_for(
                    'static', filename=f'uploads/{filename}'))
        except Exception as e:
            app.logger.error(f'Error processing image: {e}')
            return render_template(
                'result.html', prediction='Error in processing image', 
                image_url=None)

    return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(debug=False)
