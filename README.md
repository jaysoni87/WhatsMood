# 🤖 WhatsMood (aka WhatsInsight)

WhatsMood is a multilingual behavioral intelligence engine that transforms raw WhatsApp chat exports into actionable emotional and group-dynamics insights.

Unlike basic sentiment analysis tools, WhatsMood is built for real-world, messy conversations — handling code-mixed languages (Hinglish, Gujlish), Gen Z sarcasm, emoji-heavy humor, and producing executive-level summaries using Generative AI.

---

## ✨ Why WhatsMood?

Most NLP tools fail when chats include Romanized Indian languages, sarcasm like “I’m dead 💀”, emojis used for laughter (😭), and massive chat volumes that trigger API rate limits.  
WhatsMood is designed specifically to handle these challenges.

---

## 🚀 Core Capabilities

### ⚡ Daily Aggregation Engine
- Groups messages by author and day
- Reduces LLM API calls by ~92%
- Cuts processing time from ~90 minutes to ~7 minutes

### 🌍 Polyglot & Code-Mixed NLP
- Supports English, Hinglish, and Gujlish
- Detects language-specific keywords (kya, su, kem, etc.)
- Automatically translates before emotion analysis

### 🧠 Gen Z Emotion Heuristics
- Custom logic overrides standard emotion models
- Correctly decodes modern slang and emoji usage  
  - 😭 + lol → Amusement  
  - 💀 → Laughter / Shock

### 🤖 AI Executive Summaries
- Powered by Google Gemini Pro
- Generates concise 3-sentence executive summaries
- Designed for group admins and managers

### 📊 Psychometric Visualizations
- Emotion Radar Charts
- Activity Heatmaps
- Emoji–Sentiment Matrices
- Word Clouds

---

## 🛠️ Tech Stack

Language: Python 3.10+  
Dashboard: Streamlit  
Data Processing: Pandas, NumPy  
Emotion Model: RoBERTa (roberta-base-go_emotions)  
Generative AI: Google Gemini API  
Translation: Deep Translator (Google Translate)  
Visualization: Plotly Express, WordCloud  

---

## 📂 Project Structure

WhatsMood/  
├── Home.py                  (Main Streamlit dashboard)  
├── processor.py             (Core NLP, heuristics & logic engine)  
├── pages/  
│   └── 1_📊_Deep_Dive.py     (Advanced analytics & visualizations)  
├── requirements.txt         (Python dependencies)  
├── chat.txt                 (Ignored: WhatsApp chat export)  
├── .streamlit/              (Ignored: API secrets)  
│   └── secrets.toml  
└── venv/                    (Ignored: virtual environment)  

---

## ⚙️ Installation & Setup

1. Clone the repository

git clone https://github.com/YOUR_USERNAME/WhatsMood.git  
cd WhatsMood  

2. Create and activate a virtual environment

python -m venv venv  

Windows:  
venv\Scripts\activate  

Mac / Linux:  
source venv/bin/activate  

3. Install dependencies

pip install -r requirements.txt  

If requirements.txt does not exist yet, generate it using:

pip freeze > requirements.txt  

---

## 🔑 API Configuration

WhatsMood uses Google Gemini Pro for executive summaries.

Create a folder named .streamlit in the project root and add a file named secrets.toml.

Inside secrets.toml, add:

GOOGLE_API_KEY = "your_actual_api_key_here"

(This folder is gitignored and should never be committed.)

---

## 🏃‍♂️ How to Run

1. Export a WhatsApp chat without media as a .txt file  
2. Start the Streamlit app:

streamlit run Home.py  

3. Upload the chat file in the dashboard  
4. Explore behavioral insights

---

## 🧠 Processing Pipeline

1. Universal Parser  
   - Handles Android and iOS formats  
   - Cleans multiline messages  
   - Normalizes timestamps  

2. Daily Aggregation  
   - Compresses chats into daily user blocks  
   - Prevents LLM rate-limit issues  

3. Multilingual Routing  
   - Detects Hinglish and Gujlish  
   - Translates text before emotion inference  

4. Emotion Analysis  
   - RoBERTa emotion classification  
   - Gen Z heuristic overrides  

5. AI Summary Generation  
   - Structured metrics sent to Gemini Pro  
   - Executive-friendly summaries  

---

## 🔮 Future Roadmap

- Conflict detection between specific user pairs  
- RAG-based “Chat with Your Data”  
- Topic and entity extraction  
- Long-term mood trend analysis  

---

## 📝 License

This project is open-source and licensed under the MIT License.

---

## 🙌 Author Note

WhatsMood bridges the gap between raw human conversation and actionable behavioral intelligence, built specifically for multilingual, emoji-rich, real-world chats.
