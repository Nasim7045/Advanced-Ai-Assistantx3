import os
import google.generativeai as genai
import fitz  # PyMuPDF for PDFs
import streamlit as st
import pandas as pd
import docx
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import re
import pytesseract  # OCR for images
import webbrowser  # For search automation

# Set up Google Generative AI API
genai.configure(api_key="Fixed issue of prompt in Ai assistant")

# Configure generation settings
generation_config = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Initialize AI Model
model = genai.GenerativeModel("gemini-1.5-flash", generation_config=generation_config)
chat_session = model.start_chat(history=[])

# Initialize BLIP Image Captioning Model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
device = "cuda" if torch.cuda.is_available() else "cpu"
caption_model = caption_model.to(device)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        pdf_document = fitz.open(stream=pdf_file, filetype="pdf")
        for page in pdf_document:
            text += page.get_text("text") + "\n"
    except Exception as e:
        st.error(f"PDF extraction error: {e}")
    return text.strip()

# Function to extract text from Word document
def extract_text_from_word(docx_file):
    try:
        doc = docx.Document(docx_file)
        return "\n".join([para.text for para in doc.paragraphs]).strip()
    except Exception as e:
        st.error(f"Word extraction error: {e}")
        return ""

# Function to extract data from Excel file
def extract_text_from_excel(excel_file):
    try:
        df_dict = pd.read_excel(excel_file, sheet_name=None)  # Load all sheets
        return df_dict  # Dictionary of DataFrames {sheet_name: dataframe}
    except Exception as e:
        st.error(f"Excel extraction error: {e}")
        return {}

# Function to clean text
def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

# Image captioning function
def generate_image_caption(image):
    try:
        image = image.resize((224, 224)).convert("RGB")
        inputs = processor(image, return_tensors="pt").to(device)
        outputs = caption_model.generate(**inputs, max_length=50, num_beams=5)
        return processor.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        return f"Caption error: {e}"

# OCR function
def extract_image_text(image):
    try:
        image = image.resize((800, 600))  # Resize to speed up OCR
        return pytesseract.image_to_string(image)
    except Exception as e:
        return f"OCR error: {e}"

# Process image-related questions
def process_image_question(caption, ocr_text, question):
    input_text = f"Image Description: {caption}\nExtracted Text: {ocr_text}\nQuestion: {question}"
    response = chat_session.send_message(input_text)
    return response.text

# Search Automation Functions
def search_youtube(query):
    webbrowser.open(f"https://www.youtube.com/results?search_query={query}")

def search_google(query):
    webbrowser.open(f"https://www.google.com/search?q={query}")

def search_instagram(username):
    webbrowser.open(f"https://www.instagram.com/{username.replace('@', '')}/")

# Streamlit UI
st.title("Enhanced Multi-functional AI Assistant")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Documents (PDF, Word)", "Excel", "Images", "Prompting", "Search Automation"])

# Document Section (PDF & Word)
with tab1:
    st.header("Document Analysis (PDF & Word)")
    uploaded_doc = st.file_uploader("Upload PDF or Word Document", type=["pdf", "docx"])
    if uploaded_doc:
        file_extension = uploaded_doc.name.split(".")[-1].lower()
        extracted_text = extract_text_from_pdf(uploaded_doc.read()) if file_extension == "pdf" else extract_text_from_word(uploaded_doc)
        if extracted_text:
            st.session_state['doc_text'] = extracted_text
            st.success(f"{uploaded_doc.name} processed successfully!")
            if st.checkbox("Show extracted text"):
                st.text_area("Extracted Text", extracted_text, height=300)
            question = st.text_input("Ask about the document:")
            if question:
                response = chat_session.send_message(f"Document Content:\n{st.session_state['doc_text']}\n\nQuestion: {question}")
                st.write("*Response:*", response.text)

# Excel Section
with tab2:
    st.header("Excel Sheet Analysis")
    uploaded_excel = st.file_uploader("Upload Excel File", type=["xls", "xlsx"])

    if uploaded_excel:
        excel_data = extract_text_from_excel(uploaded_excel)

        if excel_data:
            sheet_names = list(excel_data.keys())
            selected_sheet = st.selectbox("Select a sheet", sheet_names)
            df = excel_data[selected_sheet]

            st.write(f"*Preview of {selected_sheet}:*")
            st.dataframe(df)  # Display table

            question = st.text_input("Ask about this sheet:")
            if question:
                input_text = f"Excel Sheet Data:\n{df.to_string(index=False)}\n\nQuestion: {question}"
                response = chat_session.send_message(input_text)
                st.write("*Response:*", response.text)

# Image Section
with tab3:
    st.header("Image Analysis")
    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, use_column_width=True)
        if 'prev_image' not in st.session_state or st.session_state['prev_image'] != uploaded_image.name:
            st.session_state['image_data'] = {}
            st.session_state['prev_image'] = uploaded_image.name
        if not st.session_state.get('image_data'):
            with st.spinner("Analyzing image..."):
                caption = generate_image_caption(image)
                ocr_text = clean_text(extract_image_text(image))
                st.session_state['image_data'] = {'caption': caption, 'ocr_text': ocr_text}
        data = st.session_state['image_data']
        col1, col2 = st.columns(2)
        with col1:
            st.write("*Visual Analysis:*", data['caption'])
        with col2:
            st.write("*Extracted Text:*", data['ocr_text'] if data['ocr_text'] else "No text detected")
        image_question = st.text_input("Ask about this image:")
        if image_question:
            response = process_image_question(data['caption'], data['ocr_text'], image_question)
            st.write("*Response:*", response)

# General AI Chat
with tab4:
    prompt = st.text_input("Ask anything:")
    if prompt:
        response = chat_session.send_message(prompt)
        st.write("*Response:*", response.text)

# Search Automation
with tab5:
    st.header("Search Automation")
    query = st.text_input("Enter the Username or your query")

    if st.button("Search on YouTube"):
        search_youtube(query)

    if st.button("Search on Instagram"):
        search_instagram(query)

    if st.button("Search on Google"):
        search_google(query)