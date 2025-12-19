import streamlit as st
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import re

# --- MODEL LOADING ---
@st.cache_resource
def load_models():
    # Keep your existing classifiers
    emotion_classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")
    sentiment_classifier = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
    
    # Load Mistral-7B-v0.1 with 4-bit quantization
    model_id = "mistralai/Mistral-7B-v0.1"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    mistral_model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    return emotion_classifier, sentiment_classifier, mistral_model, tokenizer

emotion_classifier, sentiment_classifier, mistral_model, tokenizer = load_models()

# --- UTILITIES ---
def split_into_sentences(text):
    text = re.sub(r'\s+', ' ', text.strip())
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
    return [s.strip() for s in sentences if s.strip()]

def get_agent_response(emotions, drift_score, sentiment, original_text):
    # Constructing the Prompt for the Agent
    prompt = f"""<s>[INST] You are an Emotional Intelligence AI Agent. 
    Analyze the following data and provide a brief, empathetic psychological insight.
    
    Data:
    - Text: "{original_text}"
    - Emotional Sequence: {", ".join(emotions)}
    - Drift Score: {drift_score:.2f}
    - Overall Sentiment: {sentiment}

    Provide a 2-sentence summary of the user's state of mind and one piece of advice. [/INST]"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = mistral_model.generate(**inputs, max_new_tokens=150, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the response part after the instruction
    return response.split("[/INST]")[-1].strip()

# --- STREAMLIT UI ---
st.title("🧠 AI Agent: Emotion Drift & Insights")

input_text = st.text_area("Enter your text:", height=200)

if st.button("Analyze & Consult Agent"):
    if input_text:
        sentences = split_into_sentences(input_text)
        
        if sentences:
            # 1. Classification
            emotions = [emotion_classifier(s)[0]['label'] for s in sentences]
            overall_sentiment = sentiment_classifier(input_text)[0]['label']
            
            # 2. Drift Calculation
            num_changes = sum(1 for i in range(1, len(emotions)) if emotions[i] != emotions[i-1])
            drift_score = num_changes / (len(emotions) - 1) if len(emotions) > 1 else 0.0
            
            # 3. Agent Logic (Mistral Generation)
            with st.spinner("Agent is thinking..."):
                agent_insight = get_agent_response(emotions, drift_score, overall_sentiment, input_text)

            # 4. Display Results
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Analysis Metrics")
                st.write(f"**Sentiment:** {overall_sentiment.upper()}")
                st.write(f"**Drift Score:** {drift_score:.2f}")
                
            with col2:
                st.subheader("Agent Insights")
                st.info(agent_insight)

            st.subheader("Timeline")
            st.table({"Sentence": sentences, "Emotion": emotions})
    else:
        st.warning("Please enter some text.")
