
# Landslide Detection Using Hybrid Model: ResNet50 + VGG19

This project demonstrates a **Landslide Detection** system using a **Hybrid Deep Learning Model** combining **ResNet50** and **VGG19** architectures. The goal is to identify and classify landslide-prone regions from satellite imagery, enhancing environmental monitoring and disaster management capabilities.

## Key Features

- **Hybrid Architecture:** The project combines the power of ResNet50 for residual learning and VGG19 for fine-grained feature extraction. This hybrid approach improves the accuracy of detecting landslides in images.
- **Pre-trained Models:** Utilizes pre-trained models on large datasets and fine-tunes them on a custom dataset consisting of satellite imagery.
- **Image Preprocessing:** Includes image resizing, normalization, and augmentation to ensure better model performance and generalization.
- **Evaluation Metrics:** Performance is evaluated using standard metrics such as **accuracy**, **precision**, **recall**, and **F1-score**.

## Technologies Used

- **Programming Language:** Python
- **Deep Learning Frameworks:** TensorFlow, Keras
- **Libraries:** NumPy, Matplotlib, OpenCV
- **Development Environment:** Jupyter Notebook
- **Web Framework (if applicable):** Flask (for real-time detection via a web app)

## Project Structure

The repository contains the following files and directories:

- `app.py`: Flask-based web application for real-time landslide detection.
- `landslide.ipynb`: Jupyter Notebook that contains the model training, evaluation, and testing pipeline.
- `models/`: Directory where the trained models (ResNet50 and VGG19) are stored.
- `static/`: Directory for static files such as images.
- `templates/`: Directory containing HTML templates for the web interface.
- `requirements.txt`: A list of required Python packages.

## Installation Instructions

Follow the steps below to get started with this project:

### 1. Clone the repository

```bash
git clone https://github.com/Vejandlachakrish/LandSlide-Detection-Using-Hybridmodel-Resnet50-VGG19.git

### 2. Navigate to the project directory

```bash
cd LandSlide-Detection-Using-Hybridmodel-Resnet50-VGG19
```

### 3. Install the required dependencies

Before you run the application, you need to install the required Python dependencies. You can do this by running:

```bash
pip install -r requirements.txt
```

### 4. Run the application

To launch the web application for real-time landslide detection, run:

```bash
python app.py
```

The application will be available at [http://localhost:5000](http://localhost:5000).

## How It Works

1. **Data Collection:** The project uses satellite imagery (both images with and without landslides).
2. **Image Preprocessing:** Each image is resized, normalized, and augmented for better model performance.
3. **Model Training:** The ResNet50 and VGG19 models are fine-tuned on the dataset, combining their strengths for accurate landslide detection.
4. **Model Evaluation:** The models are evaluated using key metrics, and the best-performing model is selected for deployment.
5. **Real-Time Detection:** Using the Flask app, users can upload images of terrains, and the model will predict whether a landslide is detected in the image.

## Evaluation Metrics

- **Accuracy:** The overall accuracy of the model.
- **Precision:** The model's ability to correctly identify landslide images.
- **Recall:** The model's ability to identify all actual landslide images.
- **F1-Score:** The balance between precision and recall.

## Contributing

If you would like to contribute to this project, feel free to fork the repository, create a new branch, and submit a pull request with your proposed changes.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
```
