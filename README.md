# SmartTalent-AI--Interview-Agent
SmartTalent AI: An Adaptive and Contextual Interview Agent for Intelligent Talent Screening
# -*- coding: utf-8 -*-
"""
# 🚀 AI Interview Agent 🤖

<div align="center">
  <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExcDZ1dWx5ZzV5dWg3b3lqZzR4c2R2eWJ6dWx0bGJqZzB0eGZ3eWZ6biZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/L1R1tvI9svkIWwpVYr/giphy.gif" width="300" alt="AI Interview Bot">
</div>

## 🌟 Features
- 🧠 GPT-4 powered contextual questioning
- 📄 Processes job descriptions, resumes & company profiles
- 💬 Maintains conversational memory
- 🎨 Beautiful Streamlit interface
- ⚡ Retrieval-Augmented Generation

## 🛠️ Requirements
```python
streamlit==1.32.0
langchain==0.1.0
langchain-community==0.0.11
langchain-openai==0.0.2
langchain-text-splitters==0.0.1
faiss-cpu==1.7.4
openai==1.3.0
🚀 Quick Start
Install requirements: pip install -r requirements.txt

Set OpenAI API key in the app

Upload documents and start interviewing!

"""

import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

Custom CSS for an extraordinary UI
st.markdown("""
<style>
:root {
--primary: #4a6fa5;
--secondary: #166088;
--accent: #4fc3f7;
--light: #f8f9fa;
--dark: #2c3e50;
}

    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8eb 100%);
    }
    
    .header {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        color: white;
        padding: 2rem;
        border-radius: 0 0 15px 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        color: white;
        border-radius: 12px;
        padding: 0.7rem 1.5rem;
        border: none;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    .chat-message-user {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1.2rem;
        border-radius: 18px 18px 0 18px;
        margin: 0.8rem 0;
        border-left: 5px solid var(--primary);
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .chat-message-assistant {
        background: linear-gradient(135deg, #f1f8e9 0%, #dcedc8 100%);
        padding: 1.2rem;
        border-radius: 18px 18px 18px 0;
        margin: 0.8rem 0;
        border-left: 5px solid #7cb342;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .stFileUploader>div>div>div>button {
        border-radius: 12px !important;
        padding: 0.7rem 1.5rem !important;
    }
    
    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #c8e6c9 100%);
        color: #155724;
        padding: 1.2rem;
        border-radius: 12px;
        margin: 1.2rem 0;
        border-left: 5px solid #28a745;
    }
    
    .error-box {
        background: linear-gradient(135deg, #f8d7da 0%, #ffcdd2 100%);
        color: #721c24;
        padding: 1.2rem;
        border-radius: 12px;
        margin: 1.2rem 0;
        border-left: 5px solid #dc3545;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
    }
</style>
""", unsafe_allow_html=True)

