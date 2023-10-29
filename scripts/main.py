# app.py

import streamlit as st
import torch
import re
import io
import pandas as pd
from PIL import Image
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor

# Initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Pix2StructForConditionalGeneration.from_pretrained('google/deplot').to(device)
processor = Pix2StructProcessor.from_pretrained('google/deplot')

# Define the deplot function and its helper function

def display_deplot_output(deplot_output):
    deplot_output = deplot_output.replace("<0x0A>", "\n").replace(" | ", "\t")

    second_a_index = [m.start() for m in re.finditer('\t', deplot_output)][1]
    last_newline_index = deplot_output.rfind('\n', 0, second_a_index) 

    title = deplot_output[:last_newline_index]
    table = deplot_output[last_newline_index+1:]

    st.write(title)

    data = io.StringIO(table)
    df = pd.read_csv(data, sep='\t')
    st.table(df)

def deplot(path, model, processor, device):
    image = Image.open(path)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    inputs = processor(images=image, text="Generate underlying data table of the figure below:", return_tensors="pt")

    # Move inputs to GPU
    inputs = {key: value.to(device) for key, value in inputs.items()}

    predictions = model.generate(**inputs, max_new_tokens=512)
    return processor.decode(predictions[0], skip_special_tokens=True)

# Streamlit App UI
st.title('Graph Reader App')
st.write(
    """
    Welcome to the Graph Reader App! 🚀 
    Upload an image of a graph, and the app will process it to retrieve the relevant data.
    """
)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    result = deplot(uploaded_file, model, processor, device)
    display_deplot_output(result)

st.write("Developed with ❤️ by [Your Name or Organization]")

