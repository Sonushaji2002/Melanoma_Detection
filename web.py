import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Set page config
st.set_page_config(page_title="Melanoma Detection", layout="wide")

# Load trained model
MODEL_PATH = r"D:\Skin_Cancer\model.keras"  # Update path if needed
model = load_model(MODEL_PATH)

# Define categories
CATEGORIES = ["Benign", "Malignant"]

def predict(image):
    image = image.resize((150, 150))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    
    prediction = model.predict(image)[0]  # Softmax returns array of probabilities
    class_index = np.argmax(prediction)  # Get index of highest probability
    confidence = prediction[class_index] * 100  # Convert to percentage
    label = CATEGORIES[class_index]
    
    return label, confidence

# Initialize session state for navigation
if "page" not in st.session_state:
    st.session_state.page = "home"

# Navigation Functions
def go_to_home():
    st.session_state.page = "home"

def go_to_prediction():
    st.session_state.page = "prediction"

def go_to_info():
    st.session_state.page = "info"

# Display Navigation
if st.session_state.page == "home":
    st.title("Welcome to Melanoma Detection App")
    st.write("This application helps in detecting melanoma from images.")
    if st.button("Go to Prediction 🔍"):
        go_to_prediction()
    if st.button("More Info ℹ️"):
        go_to_info()

elif st.session_state.page == "prediction":
    st.title("Melanoma Detection 🖼️")
    st.markdown("[🏠 Home](#)", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload a skin lesion image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        resized_image = image.resize((150, 150))

        st.image(resized_image, caption="Uploaded Image (Resized)", width=200)
        if st.button("Predict 🔍"):
            label, confidence = predict(resized_image)
            st.write(f"**Prediction:** {label}")
            st.write(f"**Confidence:** {confidence:.2f}%")

    st.info("This app is for educational purposes only. Consult a dermatologist for medical advice.")
    if st.button("Back to Home"):
        go_to_home()

elif st.session_state.page == "info":
    st.title("Melanoma Information")
    st.markdown("[🏠 Home](#)", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)

    with col1:
        if st.button("About Melanoma 🧬"):
            st.subheader("What is Melanoma?")
            st.write("Melanoma is a type of skin cancer that develops in melanocytes, the cells that produce melanin.")
            st.write("It is more aggressive than other skin cancers and can spread rapidly if not detected early.")

    with col2:
        if st.button("Prevention & Symptoms ⚠️"):
            st.subheader("Symptoms of Melanoma")
            st.write("- A new or changing mole with irregular borders, uneven colors, or larger than 6mm.")
            st.write("- Moles that bleed, itch, or change in size and shape.")
            st.write("- Dark spots on the skin that spread.")
            
            st.subheader("How to Prevent Melanoma")
            st.write("- Avoid excessive sun exposure and use sunscreen with SPF 30+.")
            st.write("- Regularly examine your skin for unusual moles or changes.")
            st.write("- Consult a dermatologist for regular check-ups.")
    
    if st.button("Back to Home"):
        go_to_home()
