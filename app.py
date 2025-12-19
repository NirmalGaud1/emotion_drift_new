import streamlit as st
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import re

# --- AI AGENT SETUP ---
@st.cache_resource
def load_agent_components():
    # Detect emotions and sentiment (smaller models)
    emo_pipe = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")
    sent_pipe = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
    
    # Load Mistral-7B-v0.1 for the "Agent Reasoning"
    # We use 4-bit quantization to ensure it runs efficiently
    model_id = "mistralai/Mistral-7B-v0.1"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    return emo_pipe, sent_pipe, model, tokenizer

# Load everything
emotion_classifier, sentiment_classifier, mistral_model, tokenizer = load_agent_components()

# --- UTILITIES ---
def split_into_sentences(text):
    text = re.sub(r'\s+', ' ', text.strip())
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
    return [s.strip() for s in sentences if s.strip()]

def ask_agent(text, emotions, drift_score):
    # This is where the model "thinks" like an agent
    prompt = f"<s>[INST] You are an AI Emotional Intelligence Agent. Analyze this data:\n" \
             f"Text: {text}\n" \
             f"Emotions detected: {list(set(emotions))}\n" \
             f"Emotional Drift: {drift_score:.2f}\n" \
             f"Provide a brief psychological insight. [/INST]"
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = mistral_model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).split("[/INST]")[-1]

# --- STREAMLIT UI ---
st.title("🤖 Emotional Intelligence Agent")

input_text = st.text_area("What's on your mind?", height=150)

if st.button("Consult Agent"):
    if input_text:
        sentences = split_into_sentences(input_text)
        
        # 1. Run Data Analysis
        emotions = [emotion_classifier(s)[0]['label'] for s in sentences]
        num_changes = sum(1 for i in range(1, len(emotions)) if emotions[i] != emotions[i-1])
        drift = num_changes / (len(emotions) - 1) if len(emotions) > 1 else 0
        
        # 2. Agent Reasoning
        with st.spinner("Agent is analyzing your emotional patterns..."):
            insight = ask_agent(input_text, emotions, drift)
        
        # 3. Output
        st.subheader("Agent's Insight")
        st.info(insight)
        
        with st.expander("See Raw Emotional Data"):
            st.write(f"**Drift Score:** {drift:.2f}")
            st.table({"Sentence": sentences, "Emotion": emotions})
