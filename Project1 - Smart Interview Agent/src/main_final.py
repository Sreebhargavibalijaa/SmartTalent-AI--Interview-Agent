import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Custom CSS for decorations
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .stButton>button {
            background-color: #4a6fa5;
            color: white;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            border: none;
        }
        .stButton>button:hover {
            background-color: #3a5a8a;
            color: white;
        }
        .stTextInput>div>div>input {
            border-radius: 8px;
            padding: 0.5rem;
        }
        .stFileUploader>div>div>div>button {
            border-radius: 8px;
        }
        .header {
            color: #2c3e50;
            text-align: center;
            padding: 1rem;
            border-bottom: 2px solid #4a6fa5;
            margin-bottom: 1.5rem;
        }
        .chat-message-user {
            background-color: #e3f2fd;
            padding: 1rem;
            border-radius: 10px;
            margin: 0.5rem 0;
            border-left: 4px solid #4a6fa5;
        }
        .chat-message-assistant {
            background-color: #f1f8e9;
            padding: 1rem;
            border-radius: 10px;
            margin: 0.5rem 0;
            border-left: 4px solid #7cb342;
        }
        .success-box {
            background-color: #d4edda;
            color: #155724;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        .error-box {
            background-color: #f8d7da;
            color: #721c24;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "initialized" not in st.session_state:
    st.session_state.initialized = False

# Configuration
OPENAI_API_KEY = "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"

def process_uploaded_files(uploaded_files):
    """Process uploaded files with proper validation"""
    documents = {}
    for file in uploaded_files:
        try:
            content = file.getvalue().decode("utf-8")
            if not content.strip():
                st.warning(f"File {file.name} is empty")
                continue
                
            if "jd" in file.name.lower() or "job" in file.name.lower():
                documents["job_post"] = content
            elif "resum" in file.name.lower():
                documents["candidate_resume"] = content
            else:
                documents["company_profile"] = content
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")
    return documents

def initialize_rag_system(documents):
    """Initialize RAG system with correct variable mapping"""
    try:
        # Validate documents
        required_docs = ["job_post", "company_profile", "candidate_resume"]
        if not all(doc in documents for doc in required_docs):
            missing = [doc for doc in required_docs if doc not in documents]
            st.error(f"Missing documents: {', '.join(missing)}")
            return None
        
        # Process documents
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        all_chunks = []
        for content in documents.values():
            from langchain_core.documents import Document
            chunks = splitter.split_documents([Document(page_content=content)])
            if chunks:
                all_chunks.extend(chunks)
        
        if not all_chunks:
            st.error("No valid document chunks could be created")
            return None
            
        # Create vectorstore
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vectorstore = FAISS.from_documents(all_chunks, embeddings)
        
        # Correct prompt template with proper variable names
        prompt_template = """
        You are an AI interviewer. Use this context:
        {context}
        
        Conversation history:
        {chat_history}
        
        Candidate's last response:
        {question}
        
        Generate one relevant interview question:"""
        
        # Initialize components
        llm = ChatOpenAI(model="gpt-4", temperature=0.7, api_key=OPENAI_API_KEY)
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key='question',
            output_key='answer'
        )
        
        # Create chain with proper variable mapping
        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
            memory=memory,
            chain_type="stuff",
            combine_docs_chain_kwargs={
                "prompt": ChatPromptTemplate.from_template(prompt_template),
                "document_variable_name": "context"
            },
            verbose=True
        )
        
    except Exception as e:
        st.error(f"Initialization error: {str(e)}")
        return None

# Streamlit UI with decorations
st.markdown('<div class="header"><h1>🤖 AI Interview Agent</h1></div>', unsafe_allow_html=True)
st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p>Upload job post, company profile, and candidate resume to begin the interview</p>
    </div>
""", unsafe_allow_html=True)

# File uploader with decoration
with st.container():
    st.markdown("### 📁 Upload Required Documents")
    uploaded_files = st.file_uploader(
        " ",
        accept_multiple_files=True,
        type=["txt"],
        help="Upload: 1) Job Description 2) Company Profile 3) Candidate Resume"
    )

if uploaded_files and len(uploaded_files) == 3:
    if st.button("🚀 Initialize Interview", key="init_button"):
        with st.spinner("⚙️ Setting up interview system..."):
            documents = process_uploaded_files(uploaded_files)
            if documents and len(documents) == 3:
                st.session_state.chain = initialize_rag_system(documents)
                if st.session_state.chain:
                    st.session_state.initialized = True
                    st.session_state.messages = [{
                        "role": "assistant",
                        "content": "👋 Welcome! Let's begin with your relevant experience."
                    }]
                    st.markdown('<div class="success-box">✅ System ready! Start your interview.</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="error-box">❌ Failed to initialize - check document formats</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="error-box">❌ Couldn\'t process all documents</div>', unsafe_allow_html=True)

# Interview interface with decorated messages
if st.session_state.get("initialized"):
    st.markdown("### 💬 Interview Session")
    
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-message-user"><strong>You:</strong> {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message-assistant"><strong>Interviewer:</strong> {msg["content"]}</div>', unsafe_allow_html=True)
    
    if prompt := st.chat_input("Type your response..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.spinner("🤔 Generating question..."):
            try:
                response = st.session_state.chain({"question": prompt})
                ai_response = response["answer"]
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": ai_response
                })
                st.rerun()
            except Exception as e:
                st.markdown(f'<div class="error-box">⚠️ Error: {str(e)}</div>', unsafe_allow_html=True)