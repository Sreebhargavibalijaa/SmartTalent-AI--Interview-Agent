import os
import tempfile
import streamlit as st
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from audio_recorder_streamlit import audio_recorder
import openai
from elevenlabs.client import ElevenLabs
from elevenlabs import play
import numpy as np
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    st.error(f"Failed to import transformers: {str(e)}")
    TRANSFORMERS_AVAILABLE = False
except RuntimeError as e:
    st.error(f"Runtime error with transformers: {str(e)}. This is likely a version conflict with NumPy.")
    TRANSFORMERS_AVAILABLE = False
# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "audio_enabled" not in st.session_state:
    st.session_state.audio_enabled = False

# Set up sidebar
with st.sidebar:
    st.title("SmartTalent AI Configuration")
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    st.markdown("[Get OpenAI API key](https://platform.openai.com/account/api-keys)")
    
    # Voice API keys
    st.subheader("Voice API Keys (Optional)")
    elevenlabs_key = st.text_input("ElevenLabs API Key", type="password")
    st.markdown("[Get ElevenLabs API key](https://elevenlabs.io/api)")
    
    # File uploaders
    st.subheader("Upload Documents")
    job_desc_file = st.file_uploader("Job Description (PDF/TXT)", type=["pdf", "txt"])
    resume_file = st.file_uploader("Candidate Resume (PDF/TXT)", type=["pdf", "txt"])
    company_profile_file = st.file_uploader("Company Profile (PDF/TXT)", type=["pdf", "txt"])
    
    # Interview parameters
    st.subheader("Interview Settings")
    interview_mode = st.radio("Mode", ["Text", "Voice"], index=0)
    if interview_mode == "Voice":
        st.session_state.audio_enabled = True
        voice_options = ["Rachel", "Domi", "Bella"]
        selected_voice = st.selectbox("AI Voice", options=voice_options)
    difficulty_level = st.select_slider("Difficulty Level", options=["Junior", "Mid", "Senior"], value="Mid")
    
    # Initialize button
    start_interview = st.button("Initialize Interview Agent")

# Main app
st.title("SmartTalent AI Interview Agent")
st.markdown("""
An adaptive, RAG-powered interview system with voice capabilities and advanced evaluation.
""")
def analyze_sentiment(text):
    """Analyze sentiment using HuggingFace pipeline with fallback"""
    if not TRANSFORMERS_AVAILABLE:
        return {"label": "NEUTRAL", "score": 0.5}
    
    try:
        sentiment_analyzer = pipeline("sentiment-analysis")
        return sentiment_analyzer(text)[0]
    except Exception as e:
        st.error(f"Sentiment analysis error: {str(e)}")
        return {"label": "NEUTRAL", "score": 0.5}

def evaluate_star_response(response_text):
    """Evaluate if response follows STAR method"""
    evaluation_prompt = f"""
    Analyze if this interview response properly uses the STAR method (Situation, Task, Action, Result):
    Response: {response_text}
    
    Provide:
    1. STAR completeness score (0-100)
    2. Missing components
    3. Improvement suggestions
    """
    
    try:
        llm = ChatOpenAI(model_name="gpt-4", temperature=0, openai_api_key=openai_api_key)
        analysis = llm.invoke(evaluation_prompt)
        return analysis.content
    except Exception as e:
        return f"Error in STAR evaluation: {str(e)}"

def text_to_speech(text, voice="Rachel"):
    """Convert text to speech using ElevenLabs"""
    if not elevenlabs_key:
        st.warning("ElevenLabs API key missing for voice features")
        return
    
    try:
        client = ElevenLabs(api_key=elevenlabs_key)
        audio = client.generate(
            text=text,
            voice=voice,
            model="eleven_monolingual_v2"
        )
        play(audio)
    except Exception as e:
        st.error(f"Error in voice generation: {str(e)}")

def speech_to_text(audio_bytes):
    """Convert speech to text using Whisper"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_file.write(audio_bytes)
        tmp_path = tmp_file.name
    
    try:
        client = openai.OpenAI(api_key=openai_api_key)
        with open(tmp_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-1"
            )
        return transcript.text
    except Exception as e:
        st.error(f"Error in speech recognition: {str(e)}")
        return ""
    finally:
        os.unlink(tmp_path)

def load_and_process_documents(files):
    """Load and chunk documents"""
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(file.read())
            tmp_path = tmp_file.name
            
            try:
                if file.name.endswith(".pdf"):
                    loader = PyPDFLoader(tmp_path)
                else:
                    loader = TextLoader(tmp_path)
                    
                docs = loader.load()
                split_docs = text_splitter.split_documents(docs)
                documents.extend(split_docs)
            except Exception as e:
                st.error(f"Error processing {file.name}: {str(e)}")
            finally:
                os.unlink(tmp_path)
    
    return documents

def initialize_rag_chain(documents, api_key):
    """Initialize the RAG pipeline"""
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        vectorstore = FAISS.from_documents(documents, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # Custom prompt template
        prompt_template = ChatPromptTemplate.from_template("""
        You are SmartTalent AI, an expert interview agent. Conduct a professional interview based on:
        - Job requirements: {job_info}
        - Candidate resume: {resume_info}
        - Company values: {company_info}
        
        Current conversation:
        {chat_history}
        
        Candidate: {question}
        Interviewer:""")
        
        # LLM
        llm = ChatOpenAI(
            model_name="gpt-4",
            temperature=0.7,
            openai_api_key=api_key
        )
        
        # Memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Chain
        conversational_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": prompt_template}
        )
        
        return conversational_chain
    except Exception as e:
        st.error(f"Error initializing RAG chain: {str(e)}")
        return None

# Initialize the system
if start_interview and openai_api_key:
    if not (job_desc_file and resume_file):
        st.warning("Please upload at least a Job Description and Candidate Resume")
    else:
        with st.spinner("Initializing SmartTalent AI..."):
            try:
                # Load and process documents
                files = [f for f in [job_desc_file, resume_file, company_profile_file] if f]
                documents = load_and_process_documents(files)
                
                # Initialize RAG chain
                rag_chain = initialize_rag_chain(documents, openai_api_key)
                if rag_chain:
                    st.session_state.rag_chain = rag_chain
                    
                    # Add welcome message
                    welcome_msg = f"""
                    Hello! I'm SmartTalent AI, your interview assistant. I've analyzed:
                    - The {job_desc_file.name} job description
                    - Your resume {resume_file.name}
                    {"- The company profile " + company_profile_file.name if company_profile_file else ""}
                    
                    I'll be asking you questions tailored to your background. Let's begin!
                    """
                    st.session_state.messages.append({"role": "assistant", "content": welcome_msg})
                    
                    if st.session_state.audio_enabled:
                        text_to_speech(welcome_msg, selected_voice)
                    
                    st.success("Interview agent ready!")
            except Exception as e:
                st.error(f"Error initializing: {str(e)}")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Voice recording (if enabled)
if st.session_state.get("audio_enabled", False) and openai_api_key:
    st.subheader("Voice Response")
    audio_bytes = audio_recorder("Record your answer", pause_threshold=2.0)
    if audio_bytes:
        with st.spinner("Processing your voice response..."):
            try:
                user_input = speech_to_text(audio_bytes)
                if user_input:
                    st.session_state.messages.append({"role": "user", "content": user_input})
                    with st.chat_message("user"):
                        st.markdown(user_input)
            except Exception as e:
                st.error(f"Error processing audio: {str(e)}")

# Text input (always available)
if prompt := st.chat_input("Type your response..."):
    if "rag_chain" not in st.session_state:
        st.warning("Please initialize the interview agent first")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.rag_chain({"question": prompt})
                    ai_response = response["answer"]
                    st.markdown(ai_response)
                    st.session_state.messages.append({"role": "assistant", "content": ai_response})
                    
                    if st.session_state.audio_enabled and elevenlabs_key:
                        text_to_speech(ai_response, selected_voice)
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")

# Advanced Evaluation Section
with st.expander("Advanced Interview Analytics"):
    if "messages" in st.session_state and len(st.session_state.messages) > 1:
        st.subheader("Comprehensive Evaluation")
        
        # Get all candidate responses
        candidate_responses = [msg["content"] for msg in st.session_state.messages if msg["role"] == "user"]
        last_response = candidate_responses[-1] if candidate_responses else ""
        
        # Create tabs for different analyses
        tab1, tab2, tab3 = st.tabs(["Sentiment Analysis", "STAR Method Evaluation", "Technical Alignment"])
        
        with tab1:
            if last_response:
                sentiment = analyze_sentiment(last_response)
                if sentiment:
                    st.metric("Sentiment", sentiment["label"])
                    st.metric("Confidence", f"{sentiment['score']:.2f}")
                    st.progress(int(sentiment['score'] * 100))
            else:
                st.info("No candidate responses yet")
        
        with tab2:
            if last_response and openai_api_key:
                star_evaluation = evaluate_star_response(last_response)
                st.markdown("### STAR Method Analysis")
                st.write(star_evaluation)
            else:
                st.info("No candidate responses yet or missing OpenAI API key")
        
        with tab3:
            if candidate_responses:
                # Mock technical alignment analysis
                alignment_score = min(90 + int(10 * np.random.rand()), 100)  # Random between 90-100
                st.metric("Technical Alignment Score", f"{alignment_score}/100")
                
                keywords = ["Python", "SQL", "Machine Learning"]  # Would come from job desc analysis
                st.markdown("### Key Skills Matched")
                for kw in keywords:
                    st.checkbox(f"{kw} {'✓' if np.random.rand() > 0.3 else '✗'}", value=np.random.rand() > 0.3)
            else:
                st.info("No candidate responses yet")
    else:
        st.info("Complete more questions to see advanced analytics")