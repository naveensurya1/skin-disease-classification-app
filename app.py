import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image


# -----------------------------
# Load Model
# -----------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "best_skin_cancer_model.keras"
    )


model = load_model()


# -----------------------------
# Load Class Names
# -----------------------------

with open("class_names.json", "r") as f:
    class_names = json.load(f)


# Disease name mapping
disease_names = {
    "akiec": "Actinic Keratoses / Intraepithelial Carcinoma",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis-like Lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevi",
    "vasc": "Vascular Lesions"
}


# -----------------------------
# Streamlit Page
# -----------------------------

st.set_page_config(
    page_title="Skin Disease Classification",
    page_icon="🩺",
    layout="centered"
)

st.title("Skin Disease Classification")

st.write(
    "Upload a skin lesion image to classify it using "
    "a trained EfficientNetB0 model."
)

st.warning(
    "This application is for educational and research purposes "
    "only and should not be used as a medical diagnosis."
)


# -----------------------------
# Upload Image
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload a skin lesion image",
    type=["jpg", "jpeg", "png"]
)


# -----------------------------
# Prediction
# -----------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Predict"):

        # Resize image
        img = image.resize((224, 224))

        # Convert to NumPy array
        img_array = np.array(img)

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        # Model prediction
        predictions = model.predict(
            img_array,
            verbose=0
        )[0]

        # Get predicted class
        predicted_index = np.argmax(predictions)

        predicted_class = class_names[predicted_index]

        # Convert abbreviation to full name
        disease = disease_names.get(
            predicted_class,
            predicted_class
        )

        # Display result
        st.subheader("Prediction")

        st.success(disease)