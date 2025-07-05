import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from fpdf import FPDF
import tempfile
import io
import os
from datetime import datetime

st.set_page_config(page_title="Crack Detection App", layout="centered")
st.title("🧱 Crack Detection System (Positive/Negative)")
st.markdown("Upload tile image(s) to detect whether it's **Positive (Cracked)** or **Negative (Clean)**.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("crack_classifier_positive_negative.h5")

model = load_model()

def preprocess_image(img):
    img = img.resize((64, 64))
    img_array = np.array(img) / 255.0
    if img_array.ndim == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)
    img_array = img_array.reshape(1, 64, 64, 3)
    return img_array

uploaded_files = st.file_uploader("Upload image(s)...", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

results = []
positive_count = 0
negative_count = 0

if uploaded_files:
    st.subheader("🖼️ Predictions")

    for uploaded_file in uploaded_files:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption=uploaded_file.name)
        except Exception as e:
            st.error(f"Error loading image '{uploaded_file.name}': {e}")
            continue

        features = preprocess_image(image)
        prediction = model.predict(features)[0][0]

        label = "Positive" if prediction >= 0.5 else "Negative"
        confidence = prediction if label == "Positive" else 1 - prediction

        st.markdown(
            f"**Result for {uploaded_file.name}:** <span style='color:blue;'>{label}</span> "
            f"({confidence:.2f} confidence)",
            unsafe_allow_html=True
        )

        st.caption(f"Confidence: {confidence * 100:.1f}%")
        st.progress(int(confidence * 100))

        results.append((image, uploaded_file.name, label))
        if label == "Positive":
            positive_count += 1
        else:
            negative_count += 1

    if positive_count + negative_count > 0:
        st.subheader("📊 Summary")
        fig, ax = plt.subplots()
        ax.pie([positive_count, negative_count],
               labels=["Positive (Cracked)", "Negative (Clean)"],
               autopct='%1.1f%%',
               colors=["red", "green"])
        ax.axis("equal")
        st.pyplot(fig)
        pie_path = "pie_chart.png"
        fig.savefig(pie_path)

    if st.button("📄 Generate PDF Report"):
        with st.spinner("Generating PDF..."):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=14)
            pdf.cell(200, 10, "Crack Detection Report", ln=True, align='C')
            pdf.set_font("Arial", size=10)
            pdf.cell(200, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
            pdf.image(pie_path, x=10, y=30, w=180)

            for img, name, label in results:
                pdf.add_page()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmpfile:
                    img.save(tmpfile.name, "JPEG")
                    pdf.set_font("Arial", size=12)
                    pdf.cell(200, 10, f"Image: {name}", ln=True)
                    pdf.cell(200, 10, f"Prediction: {label}", ln=True)
                    pdf.image(tmpfile.name, x=10, y=30, w=80)

            pdf_bytes = pdf.output(dest='S').encode('latin1')
            st.download_button("📄 Download PDF Report", data=io.BytesIO(pdf_bytes),
                               file_name="crack_report.pdf", mime="application/pdf")

        os.remove(pie_path)
