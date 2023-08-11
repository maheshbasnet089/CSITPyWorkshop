import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

# Load the pre-trained face detection classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Function to extract face features from a single image
def extract_face_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
    
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        face_region = cv2.resize(image[y:y+h, x:x+w], (224, 224))
        return face_region.flatten()
    else:
        return None

# Paths to male and female image folders
male_image_folder = 'man/'
female_image_folder = 'woman/'

# Collect image paths from the folders
male_image_paths = [os.path.join(male_image_folder, image) for image in os.listdir(male_image_folder)]
female_image_paths = [os.path.join(female_image_folder, image) for image in os.listdir(female_image_folder)]

# Extract features and create labels for male and female images
male_features = [extract_face_features(image_path) for image_path in male_image_paths]
female_features = [extract_face_features(image_path) for image_path in female_image_paths]

# Remove None features and corresponding labels
male_features = [features for features in male_features if features is not None]
female_features = [features for features in female_features if features is not None]

# Create labels for male and female features
male_labels = [0] * len(male_features)  # 'male' label is mapped to 0
female_labels = [1] * len(female_features)  # 'female' label is mapped to 1

# Combine features and labels
all_features = male_features + female_features
all_labels = male_labels + female_labels

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(all_features, all_labels, test_size=0.2, random_state=42)

# Train an SVM classifier
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
classifier = SVC(kernel='linear')
classifier.fit(X_train_scaled, y_train)

# Predict the gender of the image
def predict_gender(image_features):
    scaled_features = scaler.transform([image_features])
    predicted_label = classifier.predict(scaled_features)
    return "male" if predicted_label[0] == 0 else "female"

# Path to the image you want to predict
image_to_predict = 'fe.jpg'

# Extract features from the image to predict
image_features = extract_face_features(image_to_predict)

if image_features is not None:
    # Predict the gender of the image
    predicted_gender = predict_gender(image_features)
    print("Predicted Gender:", predicted_gender)
else:
    print("No face detected in the image.")
