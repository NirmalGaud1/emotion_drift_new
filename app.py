import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import re
import torch  # For device management
from langchain_community.llms import HuggingFacePipeline
from langchain_core.tools import tool
from langchain.agents import initialize_agent, AgentType
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

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

# Define tools for the agent (same as before)
@tool
def split_sentences(text: str) -> list:
    """Split input text into sentences."""
    text = re.sub(r'\s+', ' ', text.strip())
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
    return [s.strip() for s in sentences if s.strip()]

@tool
def classify_emotions(sentences: list) -> list:
    """Classify emotions for a list of sentences."""
    return [emotion_classifier(s)[0]['label'] for s in sentences]

@tool
def compute_drift_score(emotions: list) -> float:
    """Compute emotion drift score from a list of emotions."""
    if len(emotions) <= 1:
        return 0.0
    num_changes = sum(1 for i in range(1, len(emotions)) if emotions[i] != emotions[i-1])
    return num_changes / (len(emotions) - 1)

@tool
def get_overall_sentiment(text: str) -> str:
    """Get overall sentiment for the input text."""
    return sentiment_classifier(text)[0]['label'].upper()

# Load Mistral-7B-v0.1 locally (no API, no Ollama)
model_id = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
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

# Wrap in LangChain's HuggingFacePipeline
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Initialize agent with tools
tools = [split_sentences, classify_emotions, compute_drift_score, get_overall_sentiment]
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True  # For debugging; set to False in production
)

# Custom prompt for agent to focus on emotion analysis
prompt_template = PromptTemplate(
    input_variables=["input"],
    template="You are an emotion drift analysis agent. Analyze the user's query about text emotions, drift, or sentiment. Use tools to process. Query: {input}"
)

chain = LLMChain(llm=llm, prompt=prompt_template)

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
        # Run agent through chain
        prompted_input = prompt_template.format(input=user_input)
        response = agent.run(prompted_input)
        
        # Post-process response (e.g., format timeline if detected)
        st.markdown(response)
        
        # Optional: If response mentions emotions, visualize (agent can describe, but we can enhance)
        if "emotions" in response.lower():
            # Example: Extract and show emoji timeline (agent handles logic, but demo here)
            pass  # Add custom visualization if needed

    st.session_state.messages.append({"role": "assistant", "content": response})
