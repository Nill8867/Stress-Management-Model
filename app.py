import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Load the trained model
model = tf.keras.models.load_model('stressdetect.h5')

# Define a function to preprocess images for the model
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    image = cv2.resize(image, (48, 48))  # Resize to match model input size
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    return image

# Define a function to make predictions
def predict_stress(image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    stress_prob = predictions[0][1]  # Probability of stress
    return stress_prob

# Define a custom video transformer for real-time stress detection
class StressDetectionTransformer(VideoTransformerBase):
    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")  # Convert frame to ndarray
        processed_frame = preprocess_image(image)
        
        # Predict stress
        stress_prob = model.predict(processed_frame)[0][1]
        stress_percentage = stress_prob * 100

        # Display stress percentage on the video frame
        text = f"Stress Level: {stress_percentage:.2f}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, text, (10, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

        return image

# Main function to run the Streamlit app
def main():
    st.title("Live Stress Detection App")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    mode = st.sidebar.radio("Choose a mode", ["Upload Image", "Live Camera"])

    if mode == "Upload Image":
        st.subheader("Upload an image to detect stress levels")
        uploaded_file = st.file_uploader("Choose an image...", type="jpg")
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            st.write("Classifying...")
            
            # Predict stress
            stress_prob = predict_stress(np.array(image))
            stress_percentage = stress_prob * 100
            st.write(f"Detected Stress Level: {stress_percentage:.2f}%")
    elif mode == "Live Camera":
        st.subheader("Live Stress Detection")
        st.write("Start your webcam and detect stress levels in real-time.")
        webrtc_streamer(key="example", video_transformer_factory=StressDetectionTransformer)

if _name_ == "_main_":
    main()
