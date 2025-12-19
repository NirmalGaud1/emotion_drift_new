import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import re
import ast
import torch

# --- CONFIGURATION ---
# Using a smaller model for stability, or stay with Mistral if you have 24GB+ VRAM
MODEL_ID = "mistralai/Mistral-7B-v0.1" 

@st.cache_resource
def load_models():
    # Emotion & Sentiment pipelines
    emo = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")
    sent = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
    
    # LLM Loading with optimization
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto", # Automatically chooses GPU if available
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True
    )
    
    gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.1 # Lower temperature for more consistent tool use
    )
    return emo, sent, gen_pipeline

emotion_classifier, sentiment_classifier, hf_pipeline = load_models()

# --- TOOL DEFINITIONS ---
def split_sentences(text: str) -> list:
    text = re.sub(r'\s+', ' ', text.strip())
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
    return [s.strip() for s in sentences if s.strip()]

def classify_emotions(sentences: list) -> list:
    if isinstance(sentences, str): sentences = [sentences]
    return [emotion_classifier(s)[0]['label'] for s in sentences]

def compute_drift_score(emotions: list) -> float:
    if len(emotions) <= 1: return 0.0
    num_changes = sum(1 for i in range(1, len(emotions)) if emotions[i] != emotions[i-1])
    return round(num_changes / (len(emotions) - 1), 2)

def get_overall_sentiment(text: str) -> str:
    return sentiment_classifier(text)[0]['label'].upper()

tools = {
    "split_sentences": split_sentences,
    "classify_emotions": classify_emotions,
    "compute_drift_score": compute_drift_score,
    "get_overall_sentiment": get_overall_sentiment
}

# --- AGENT LOGIC ---
REACT_PROMPT = """You are an assistant that uses tools to analyze text.
Available tools:
- split_sentences(text): Breaks text into list of sentences.
- classify_emotions(list_of_sentences): Returns list of emotions.
- compute_drift_score(list_of_emotions): Returns a 0-1 score of emotional change.
- get_overall_sentiment(text): Returns POSITIVE or NEGATIVE.

Format:
Thought: Reason about what to do.
Action: tool_name(input)
Observation: tool result
... (repeat)
Final Answer: summarize everything.

Question: {query}
"""

def parse_action(gen_text):
    # Find Action: ... and stop before the next Observation or Thought
    match = re.search(r"Action:\s*(\w+)\((.*)\)", gen_text)
    if match:
        return match.group(1), match.group(2)
    return None, None

def run_agent(query):
    prompt = REACT_PROMPT.format(query=query)
    iterations = 0
    
    while iterations < 5:
        output = hf_pipeline(prompt, stop_sequence=["Observation:"])[0]['generated_text']
        new_content = output[len(prompt):]
        prompt += new_content
        
        if "Final Answer:" in new_content:
            return new_content.split("Final Answer:")[-1].strip()
        
        tool_name, tool_input_str = parse_action(new_content)
        if tool_name in tools:
            try:
                # Safer than eval()
                clean_input = ast.literal_eval(tool_input_str)
                observation = tools[tool_name](clean_input)
                obs_text = f"\nObservation: {observation}\nThought: "
                prompt += obs_text
            except Exception as e:
                prompt += f"\nObservation: Error processing input: {e}\n"
        else:
            break
        iterations += 1
    return "The agent could not reach a final answer."

# --- STREAMLIT UI ---
st.title("🧠 Emotion Drift Agent")

if user_input := st.chat_input("Enter text for emotional analysis"):
    st.chat_message("user").write(user_input)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = run_agent(user_input)
            st.write(response)
