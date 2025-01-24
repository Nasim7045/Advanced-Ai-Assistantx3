import os
import google.generativeai as genai
import fitz  # PyMuPDF
import streamlit as st
from PIL import Image  # For image processing
from transformers import BlipProcessor, BlipForConditionalGeneration
# it is best
# Set up Google Generative AI API with a hardcoded API key
api_key = "AIzaSyAT8CCucA7l5ZfFLJNDe8X082en6M-s0EE"
genai.configure(api_key=api_key)

# Configure generation settings for the model
generation_config = {
    "temperature": 1.2,
    "top_p": 0.9,
    "top_k": 50,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Initialize the GenerativeModel with your configuration
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)
chat_session = model.start_chat(history=[])

# Initialize the BLIP model and processor for image captioning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_document = fitz.open(stream=pdf_file, filetype="pdf")
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text

# Function to generate a caption from the image
def generate_image_caption(image):
    inputs = processor(image, return_tensors="pt")
    outputs = caption_model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

# Streamlit UI with multiple options
st.title("Multi-functional AI Assistant")

# Tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["Document Analysis", "Image Recognition", "Prompting"])

# Tab 1: Document Analysis
with tab1:
    st.header("Document Analysis")
    uploaded_pdf = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_pdf:
        try:
            # Extract text from PDF
            pdf_text = extract_text_from_pdf(uploaded_pdf.read())
            st.success("PDF uploaded and text extracted successfully!")
            
            # Input for questions
            question = st.text_input("Enter your question about the PDF content:")

            if question:
                input_text = f"Here is the text from the PDF:\n\n{pdf_text}\n\nQuestion: {question}"
                response = chat_session.send_message(input_text)
                response_text = response.text

                # Display the response
                st.write("AI Response:", response_text)

                # Save the conversation history
                chat_session.history.append({"role": "user", "parts": [question]})
                chat_session.history.append({"role": "model", "parts": [response_text]})

        except Exception as e:
            st.error(f"An error occurred while processing the PDF: {e}")

# Tab 2: Image Recognition with Question-Answer Capability
with tab2:
    st.header("Image Recognition with Q&A")
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        try:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Generate a description of the image
            image_caption = generate_image_caption(image)
            st.write("Image Description:", image_caption)

            # Prompt for user questions about the image
            image_question = st.text_input("Ask a question about the image:")

            if image_question:
                # Combine image description and question
                input_text = f"Image Description: {image_caption}. Now answer this question: {image_question}"
                response = chat_session.send_message(input_text)
                response_text = response.text

                # Display the response
                st.write("AI Response:", response_text)

                # Save the conversation history
                chat_session.history.append({"role": "user", "parts": [image_question]})
                chat_session.history.append({"role": "model", "parts": [response_text]})

        except Exception as e:
            st.error(f"An error occurred while processing the image: {e}")

# Tab 3: Prompting
with tab3:
    st.header("Basic Prompting")
    user_prompt = st.text_input("Ask the AI assistant a question:")

    if user_prompt:
        try:
            response = chat_session.send_message(user_prompt)
            response_text = response.text
            st.write("AI Response:", response_text)

            # Save the conversation history
            chat_session.history.append({"role": "user", "parts": [user_prompt]})
            chat_session.history.append({"role": "model", "parts": [response_text]})

        except Exception as e:
            st.error(f"An error occurred while processing the prompt: {e}")