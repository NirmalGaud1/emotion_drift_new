import streamlit as st
import re
import torch
from transformers import pipeline

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Emotion Agent", page_icon="🧠", layout="centered")

# --- MODEL LOADING (CACHED) ---
@st.cache_resource
def load_models():
    device = 0 if torch.cuda.is_available() else -1
    
    # Emotion & Sentiment
    emotion_pipe = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", device=device)
    sentiment_pipe = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", device=device)
    
    # Text Generation (Qwen2.5-0.5B)
    gen_pipe = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct", device=device, torch_dtype="auto")
    
    return emotion_pipe, sentiment_pipe, gen_pipe

emotion_pipe, sentiment_pipe, gen_pipe = load_models()

# --- UTILS ---
emoji_map = {'joy': '😊', 'anger': '😠', 'sadness': '😢', 'fear': '😨', 'love': '❤️', 'surprise': '😲'}

def get_drift(emotions):
    if len(emotions) <= 1: return 0.0
    changes = sum(1 for i in range(1, len(emotions)) if emotions[i] != emotions[i-1])
    return round(changes / (len(emotions) - 1), 2)

# --- UI ---
st.title("🧠 AI Emotion Agent")
st.markdown("Enter your thoughts below. I will analyze your emotional flow and respond.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "analysis" in message:
            st.caption(message["analysis"])

# User Input
if user_input := st.chat_input("How are you feeling?"):
    # 1. Display User Message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Analyze
    with st.spinner("Analyzing emotional patterns..."):
        sentences = re.split(r'(?<=[.!?])\s+', user_input)
        emotions = [emotion_pipe(s)[0]['label'] for s in sentences]
        sentiment = sentiment_pipe(user_input)[0]['label']
        drift_score = get_drift(emotions)
        emoji_line = ' '.join([emoji_map.get(e, '❓') for e in emotions])
        
        analysis_text = f"Emotions: {emoji_line} | Sentiment: {sentiment} | Drift: {drift_score}"

    # 3. Generate Response
    with st.chat_message("assistant"):
        messages = [
            {"role": "system", "content": "You are a helpful, empathetic assistant. Provide a complete, short response."},
            {"role": "user", "content": user_input}
        ]
        prompt = gen_pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        gen_result = gen_pipe(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)
        reply = gen_result[0]['generated_text'].split("<|im_start|>assistant")[-1].strip().replace("<|im_end|>", "")
        
        st.markdown(reply)
        st.caption(analysis_text)
        
        # Save to history
        st.session_state.messages.append({"role": "assistant", "content": reply, "analysis": analysis_text})
