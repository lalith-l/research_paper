import streamlit as st
import json
import os
import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification


@st.cache_resource
def load_model():
    """Load the fine-tuned multi-class BERT model, tokenizer, and label map."""
    model_path = './bias_type_classifier_model'
    if not os.path.exists(model_path):
        return None, None, None
    
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    
    with open(os.path.join(model_path, 'label_map.json'), 'r') as f:
        label_map_str_keys = json.load(f)
        inverse_label_map = {int(k): v for k, v in label_map_str_keys.items()}
        
    return tokenizer, model, inverse_label_map

def save_feedback(text, actual_label_str):
    """Saves the user's text and their corrected label to a CSV file."""
    feedback_df = pd.DataFrame({'text': [text], 'label': [actual_label_str]})
    if not os.path.exists('feedback.csv'):
        feedback_df.to_csv('feedback.csv', index=False, mode='a')
    else:
        feedback_df.to_csv('feedback.csv', index=False, mode='a', header=False)

tokenizer, model, inverse_label_map = load_model()

if tokenizer is None or model is None:
    st.error("Multi-class model not found. Please run the training script.")
    st.stop()

st.title("Bias Type Classifier")
st.write(
    "This app uses a fine-tuned BERT model to predict the specific type of bias "
    "(e.g., gender, race, profession) in a piece of text."
)

if 'prediction' not in st.session_state:
    st.session_state.prediction = None

user_text = st.text_area("Enter text to analyze:", "Example: There are many male CEOs in the world.")

if st.button("Analyze"):
    if user_text:
        inputs = tokenizer(user_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            prediction_idx = torch.argmax(logits, dim=-1).item()
        
        st.session_state.prediction = prediction_idx
        st.session_state.analyzed_text = user_text
    else:
        st.info("Please enter some text to analyze.")
        st.session_state.prediction = None

if st.session_state.prediction is not None:
    predicted_label = inverse_label_map[st.session_state.prediction]
    
    st.subheader("Analysis Result")
    st.info(f"Predicted Bias Type: **{predicted_label.capitalize()}**")
    st.write("---")

    with st.form("feedback_form"):
        st.write("Was this prediction correct?")
        
        is_correct = st.radio(
            "Feedback:",
            ('Yes, it was correct.', 'No, it was wrong.')
        )
        
        corrected_label = None
        if 'No' in is_correct:
            corrected_label = st.selectbox(
                'Please select the correct bias type:',
                options=sorted(list(inverse_label_map.values()))
            )
        
        submitted = st.form_submit_button("Submit Feedback")
        
        if submitted:
            final_label = predicted_label if corrected_label is None else corrected_label
            save_feedback(st.session_state.analyzed_text, final_label)
            st.success("Thank you! Your feedback has been saved.")
            st.session_state.prediction = None