import streamlit as st
import fitz  # PyMuPDF
import joblib
import joblib

model = joblib.load("../models/model.pkl")

vectorizer = joblib.load("vectorizer.pkl")

st.markdown("<h1 style='text-align:center;'>📄 Resume Classifier</h1>", unsafe_allow_html=True)
st.write("Upload a PDF resume and get the predicted job role.")

# File uploader
uploaded_file = st.file_uploader("Choose a PDF resume", type="pdf")

def extract_text_from_pdf(pdf_file):
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

if uploaded_file:
    resume_text = extract_text_from_pdf(uploaded_file)

    if resume_text.strip():
        st.subheader("Extracted Resume Text:")
        st.write(resume_text)

        # Preprocess and classify
        resume_vector = vectorizer.transform([resume_text])
        prediction = model.predict(resume_vector)[0]
        st.success(f"Predicted Job Role: {prediction}")
    else:
        st.warning("Could not extract text from this PDF.")

