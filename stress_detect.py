import os
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Check for GPU
print("Using GPU" if tf.config.list_physical_devices('GPU') else "Using CPU")

# Define dataset and model paths
base_dir = r"/Users/dhruvloriya/Desktop/PROJECT/archive" 
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
model_path = 'enhanced_stressdetect.keras'

# Map emotions to stress levels
stress_mapping = {
    'angry': 0.9,
    'disgust': 0.1,
    'fear': 0.8,
    'happy': 0.1,
    'neutral': 0.3,
    'sad': 0.7,
    'surprise': 0.2
}

def load_data_from_folder(folder_path):
    """Loads images and labels from the dataset."""
    images = []
    labels = []
    for label in os.listdir(folder_path):
        if label in stress_mapping:
            label_path = os.path.join(folder_path, label)
            for image_file in os.listdir(label_path):
                image_path = os.path.join(label_path, image_file)
                try:
                    image = Image.open(image_path).convert('L')  # Convert to grayscale
                    image = image.resize((48, 48))
                    images.append(np.array(image))
                    labels.append(stress_mapping[label])
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")
    return np.array(images), np.array(labels)

# Check if model exists
# Check if model exists
if os.path.exists(model_path):
    print("Loading existing model...")
    model = load_model(model_path)

    # Ensure dataset is loaded for evaluation
    print("Loading dataset for evaluation...")
    X_test, y_test = load_data_from_folder(test_dir)
    X_test = X_test / 255.0  # Normalize data
    X_test = X_test.reshape(-1, 48, 48, 1)
    y_test = np.array(y_test)
else:
    print("No pre-trained model found. Training a new model...")

    # Load training and testing data
    X_train, y_train = load_data_from_folder(train_dir)
    X_test, y_test = load_data_from_folder(test_dir)

    # Normalize and preprocess data
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    X_train = X_train.reshape(-1, 48, 48, 1)
    X_test = X_test.reshape(-1, 48, 48, 1)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Split training data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Define the model
    input_image = Input(shape=(48, 48, 1))
    conv1 = Conv2D(32, (3, 3), activation='relu')(input_image)
    pool1 = MaxPooling2D((2, 2))(conv1)
    conv2 = Conv2D(64, (3, 3), activation='relu')(pool1)
    pool2 = MaxPooling2D((2, 2))(conv2)
    conv3 = Conv2D(128, (3, 3), activation='relu')(pool2)
    pool3 = MaxPooling2D((2, 2))(conv3)
    flat = Flatten()(pool3)

    dense1 = Dense(128, activation='relu')(flat)
    dropout = Dropout(0.5)(dense1)
    output = Dense(1, activation='sigmoid')(dropout)

    model = Model(inputs=input_image, outputs=output)
    model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['mae'])

    # Train the model
    print("Training the model...")
    model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=32)

    # Save the model
    model.save(model_path)
    print(f"Model saved as {model_path}")

# Test the model on the test set
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=2)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test MAE: {test_mae:.4f}")

# Real-time detection using OpenCV
print("Starting real-time stress detection...")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)  # Open the default camera
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = gray_frame[y:y + h, x:x + w]
        face_resized = cv2.resize(face, (48, 48)) / 255.0
        face_resized = np.expand_dims(face_resized, axis=-1)
        face_resized = np.expand_dims(face_resized, axis=0)

        # Predict stress level
        stress_level = model.predict(face_resized)[0][0]
        stress_category = (
            "High Stress" if stress_level > 0.6 else
            "Moderate Stress" if stress_level > 0.3 else
            "Low Stress"
        )

        # Display stress level on the video feed
        label = f"{stress_category} ({stress_level:.2f})"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imshow('Stress Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()