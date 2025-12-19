import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import re

# Load emotion and sentiment models (same as original)
emotion_classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")
sentiment_classifier = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

# Emoji mapping (same as original)
emoji_map = {
    'joy': '😊',
    'anger': '😠',
    'sadness': '😢',
    'fear': '😨',
    'love': '❤️',
    'surprise': '😲'
}

# Define tool functions (without @tool)
def split_sentences(text: str) -> list:
    """Split input text into sentences."""
    text = re.sub(r'\s+', ' ', text.strip())
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
    return [s.strip() for s in sentences if s.strip()]

def classify_emotions(sentences: list) -> list:
    """Classify emotions for a list of sentences."""
    return [emotion_classifier(s)[0]['label'] for s in sentences]

def compute_drift_score(emotions: list) -> float:
    """Compute emotion drift score from a list of emotions."""
    if len(emotions) <= 1:
        return 0.0
    num_changes = sum(1 for i in range(1, len(emotions)) if emotions[i] != emotions[i-1])
    return num_changes / (len(emotions) - 1)

def get_overall_sentiment(text: str) -> str:
    """Get overall sentiment for the input text."""
    return sentiment_classifier(text)[0]['label'].upper()

# Tools dict
tools = {
    "split_sentences": split_sentences,
    "classify_emotions": classify_emotions,
    "compute_drift_score": compute_drift_score,
    "get_overall_sentiment": get_overall_sentiment
}

# Load Mistral-7B-v0.1 locally (no API, no Ollama)
model_id = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto"  # Automatically use GPU if available
)
hf_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.15
)

# Custom LLM wrapper with stripping prompt
class CustomLLM:
    def __call__(self, prompt: str) -> str:
        full = hf_pipeline(prompt)[0]['generated_text']
        return full[len(prompt):].strip()

llm = CustomLLM()

# Hardcode ReAct prompt
react_prompt = """
You are an agent that uses tools to answer questions. Use the following format:

Thought: [your reasoning]

Action: tool_name(tool_input as python literal, e.g., "text" or ["list"])

Observation: [result from tool]

... (repeat Thought/Action/Observation as needed)

Final Answer: [final response]

Available tools: split_sentences, classify_emotions, compute_drift_score, get_overall_sentiment
"""

# Parse action function
def parse_action(action_str):
    match = re.match(r'(\w+)\((.*)\)', action_str.strip(), re.DOTALL)
    if match:
        name = match.group(1)
        input_str = match.group(2)
        try:
            input_val = eval(input_str)
            return name, input_val
        except:
            return None, None
    return None, None

# Custom agent loop
def custom_agent_run(input_query):
    messages = react_prompt + "\nQuestion: " + input_query + "\n"
    max_iterations = 5
    for _ in range(max_iterations):
        generation = llm(messages)
        messages += generation
        if "Final Answer:" in generation:
            return generation.split("Final Answer:")[-1].strip()
        elif "Action:" in generation:
            action_part = generation.split("Action:")[-1].split("\n")[0].strip()
            tool_name, tool_input = parse_action(action_part)
            if tool_name and tool_name in tools:
                try:
                    observation = tools[tool_name](tool_input)
                    messages += "\nObservation: " + str(observation) + "\n"
                except Exception as e:
                    messages += "\nObservation: Error - " + str(e) + "\n"
            else:
                messages += "\nObservation: Invalid tool\n"
        else:
            messages += "\nNo action or final answer found\n"
    return "Max iterations reached"

# Streamlit chat interface
st.title("Emotion Drift AI Agent (Powered by Mistral-7B-v0.1)")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if user_input := st.chat_input("Ask about emotion analysis (e.g., 'Analyze this text: Hello, I feel great! But anxious.')"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        # Run custom agent
        response = custom_agent_run(user_input)
        
        # Post-process response (e.g., format timeline if detected)
        st.markdown(response)
        
        # Optional: If response mentions emotions, visualize (agent can describe, but we can enhance)
        if "emotions" in response.lower():
            # Example: Extract and show emoji timeline (agent handles logic, but demo here)
            pass  # Add custom visualization if needed

    st.session_state.messages.append({"role": "assistant", "content": response})
