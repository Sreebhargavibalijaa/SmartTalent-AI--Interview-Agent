# 🚀 SmartTalent AI Interview Agent 🤖

<div align="center">
  <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExcDZ1dWx5ZzV5dWg3b3lqZzR4c2R2eWJ6dWx0bGJqZzB0eGZ3eWZ6biZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/L1R1tvI9svkIWwpVYr/giphy.gif" width="400" alt="AI Interview Bot">
  
  [![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
  [![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=OpenAI&logoColor=white)](https://openai.com/)
  [![LangChain](https://img.shields.io/badge/LangChain-00A67E?style=for-the-badge)](https://www.langchain.com/)

  *Adaptive and Contextual Interview Agent for Intelligent Talent Screening*
</div>

## 🌟 Features

- 🧠 **GPT-4 Powered**: Context-aware interview questions
- 📄 **Multi-Document Analysis**: Processes job descriptions, resumes, and company profiles
- 💬 **Conversational Memory**: Maintains interview context throughout the session
- 🎨 **Beautiful Interface**: Professional Streamlit UI with responsive design
- ⚡ **RAG Architecture**: Retrieval-Augmented Generation for relevant questioning
- 🔍 **Document Insights**: Extracts key qualifications and experience matches

## 🛠️ Tech Stack

| Component        | Technology                          |
|------------------|-------------------------------------|
| Framework        | ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=Streamlit&logoColor=white) |
| LLM              | ![OpenAI](https://img.shields.io/badge/GPT4-412991?style=flat-square&logo=OpenAI&logoColor=white) |
| Vector Store     | ![FAISS](https://img.shields.io/badge/FAISS-00A67E?style=flat-square) |
| NLP Framework    | ![LangChain](https://img.shields.io/badge/LangChain-00A67E?style=flat-square) |
| Memory           | ConversationBufferMemory            |

## 📦 Requirements

```text
streamlit==1.32.0
langchain==0.1.0
langchain-community==0.0.11
langchain-openai==0.0.2
langchain-text-splitters==0.0.1
faiss-cpu==1.7.4
openai==1.3.0
python-dotenv==1.0.0


## 📦 **How it works**
graph TD
    A[Upload Documents] --> B[Text Processing]
    B --> C[Vector Embeddings]
    C --> D[FAISS Vector Store]
    D --> E[Question Generation]
    E --> F[Conversational Memory]
    F --> G[Contextual Response]
    G --> H[Evaluation Metrics]
Clone the repository:

bash
git clone https://github.com/yourusername/SmartTalent-AI-Interview-Agent.git
cd SmartTalent-AI-Interview-Agent

🤝 Contributing
Fork the repository

Create your feature branch (git checkout -b feature/AmazingFeature)

Commit your changes (git commit -m 'Add some AmazingFeature')

Push to the branch (git push origin feature/AmazingFeature)

Open a Pull Request

📜 License
Distributed under the MIT License. See LICENSE for more information.

✉️ Contact
Project Team - smarttalent-ai@example.com
Project Link: https://github.com/yourusername/SmartTalent-AI-Interview-Agent

