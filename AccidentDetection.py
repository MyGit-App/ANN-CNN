
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page Config
st.set_page_config(
    page_title="Accident Detection System",
    page_icon="🚗",
    layout="centered"
)

# Load Model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "best_pretrained_cnn.keras"
    )
    return model

model = load_model()

# Title
st.title("🚗 Accident Detection From CCTV Images")
st.markdown(
    "Upload a CCTV image and the model will predict whether an accident has occurred."
)

# Upload Image
uploaded_file = st.file_uploader(
    "Choose an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    img = image.resize((224,224))

    img_array = np.array(img)

    # Convert grayscale to RGB if needed
    if len(img_array.shape) == 2:
        img_array = np.stack(
            (img_array,)*3,
            axis=-1
        )

    img_array = img_array/255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    
prediction = model.predict(img_array)

prob = float(prediction[0][0])

st.write(prediction[0][0])

accident_prob = (1 - prob) * 100
normal_prob = prob * 100

st.subheader("Prediction Result")

if prob < 0.5:

    st.error("🚨 Accident Detected")

    st.metric(
        "Model Confidence",
        f"{accident_prob:.2f}%"
    )

else:

    st.success("✅ No Accident Detected")

    st.metric(
        "Model Confidence",
        f"{normal_prob:.2f}%"
    )

st.write("### Probabilities")

st.write(f"🚨 Accident : {accident_prob:.2f}%")
st.write(f"✅ Non Accident : {normal_prob:.2f}%")
