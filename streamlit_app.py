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
import urllib.parse
import easyocr 
import webbrowser

# Set up Google Generative AI API
genai.configure(api_key="API-key")

generation_config = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel("gemini-1.5-flash", generation_config=generation_config)
chat_session = model.start_chat(history=[])

# Initialize BLIP Caption Model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
device = "cuda" if torch.cuda.is_available() else "cpu"
caption_model = caption_model.to(device)

# Initialize EasyOCR Reader
ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

# PDF text extraction
def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        pdf_document = fitz.open(stream=pdf_file, filetype="pdf")
        for page in pdf_document:
            text += page.get_text("text") + "\n"
    except Exception as e:
        st.error(f"PDF extraction error: {e}")
    return text.strip()

# Word text extraction
def extract_text_from_word(docx_file):
    try:
        doc = docx.Document(docx_file)
        return "\n".join([para.text for para in doc.paragraphs]).strip()
    except Exception as e:
        st.error(f"Word extraction error: {e}")
        return ""

# Excel text extraction
def extract_text_from_excel(excel_file):
    try:
        df_dict = pd.read_excel(excel_file, sheet_name=None)
        return df_dict
    except Exception as e:
        st.error(f"Excel extraction error: {e}")
        return {}

# Clean text
def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

# Generate image caption
def generate_image_caption(image):
    try:
        image = image.resize((224, 224)).convert("RGB")
        inputs = processor(image, return_tensors="pt").to(device)
        outputs = caption_model.generate(**inputs, max_length=50, num_beams=5)
        return processor.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        return f"Caption error: {e}"

# OCR using EasyOCR
def extract_image_text_easyocr(image):
    try:
        image_np = image.convert("RGB")
        result = ocr_reader.readtext(np.array(image_np))
        return " ".join([text for _, text, _ in result])
    except Exception as e:
        return f"OCR error: {e}"

# Process image questions
def process_image_question(caption, ocr_text, question):
    input_text = f"Image Description: {caption}\nExtracted Text: {ocr_text}\nQuestion: {question}"
    response = chat_session.send_message(input_text)
    return response.text

# search automation
def search_youtube(query):
    webbrowser.open(f"https://www.youtube.com/results?search_query={query}")

def search_google(query):
    webbrowser.open(f"https://www.google.com/search?q={query}")

def search_instagram(username):
    webbrowser.open(f"https://www.instagram.com/{username.replace('@', '')}/")

# Streamlit UI
st.title("AX*3 Advanced Ai Assistant")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Documents (PDF, Word)", "Excel", "Images", "Prompting","Search Automation"])

# Tab 1: Documents
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

# Tab 2: Excel
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
            st.dataframe(df)
            question = st.text_input("Ask about this sheet:")
            if question:
                input_text = f"Excel Sheet Data:\n{df.to_string(index=False)}\n\nQuestion: {question}"
                response = chat_session.send_message(input_text)
                st.write("*Response:*", response.text)

# Tab 3: Images
import numpy as np  # Required for EasyOCR input

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
                ocr_text = clean_text(extract_image_text_easyocr(image))
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

# Tab 4: Prompting
with tab4:
    prompt = st.text_input("Ask anything:")
    if prompt:
        response = chat_session.send_message(prompt)
        st.write("*Response:*", response.text)


with tab5:
    st.header("Search Automation")
    
    search_query = st.text_input("Enter your search query or username:", key="search_query")
    
    if search_query:
        encoded_query = urllib.parse.quote(search_query)
        
        st.write("### Quick Search Links")
        
        youtube_url = f"https://www.youtube.com/results?search_query={encoded_query}"
        instagram_url = f"https://www.instagram.com/{search_query.replace('@', '')}/"
        google_url = f"https://www.google.com/search?q={encoded_query}"
        
        # Create columns for the search options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"[![YouTube](https://img.shields.io/badge/YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white)]({youtube_url})")
            
        with col2:
            st.markdown(f"[![Instagram](https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white)]({instagram_url})")
            
        with col3:
            st.markdown(f"[![Google](https://img.shields.io/badge/Google-4285F4?style=for-the-badge&logo=google&logoColor=white)]({google_url})")
        
        with st.expander("Direct Browser Opening (Local Only)"):
            st.warning("Note: This option only works when running the app locally on your machine.")
            
            col1, col2, col3 = st.columns(3)
            
            if col1.button("Open YouTube"):
                search_youtube(encoded_query)
                
            if col2.button("Open Instagram"):
                search_instagram(search_query)
                
            if col3.button("Open Google"):
                search_google(encoded_query)
        
        # Additional search options
        with st.expander("More search options"):
            col1, col2 = st.columns(2)
            
            twitter_url = f"https://twitter.com/search?q={encoded_query}"
            linkedin_url = f"https://www.linkedin.com/search/results/all/?keywords={encoded_query}"
            github_url = f"https://github.com/search?q={encoded_query}"
            amazon_url = f"https://www.amazon.in/s?k={encoded_query}"
            
            col1.markdown(f"[![Twitter/X](https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)]({twitter_url})")
            col1.markdown(f"[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)]({linkedin_url})")
            col2.markdown(f"[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)]({github_url})")
            col2.markdown(f"[![Amazon](https://img.shields.io/badge/Amazon-FF9900?style=for-the-badge&logo=amazon&logoColor=white)]({amazon_url})")