# 🤖 Multilingual WhatsApp Behavioral Analyzer

### 🚀 The Problem
Corporate communication analytics tools (like Slack/Teams) fail to capture the **informal, high-velocity decision-making** that happens on WhatsApp, especially in multilingual markets (India/SE Asia). They miss the "Shadow IT" where real work happens.

### 💡 The Solution
This tool is a **Semantic Engine** that transforms chaotic, code-mixed (Hindi/Gujarati/English) chat logs into **Actionable Business Intelligence**.

### 🔑 Key Capabilities
1.  **Code-Mixed NLP:** Custom routing engine to handle *Hinglish* and *Gujlish* without crashing.
2.  **Psychometric Profiling:** Daily aggregation to track team "Vibe" (Burnout vs. Joy) over time.
3.  **AI Executive Summaries:** Integrates **Google Gemini 1.5** to generate automated daily standup reports from raw logs.
4.  **Nuance Detection:** Custom heuristics to detect **Confusion** (bottlenecks) and **Sarcasm** (toxic positivity).

### 🛠️ Tech Stack
* **Engine:** Python, Pandas, Regex
* **AI/ML:** HuggingFace Transformers (RoBERTa), Google Gemini (LLM)
* **Viz:** Streamlit, Plotly, WordCloud
* **Optimization:** Batch Processing with API Rate-Limiting protection.

### 📉 Impact
* **For HR:** Identifies burnout risks by tracking late-night activity spikes.
* **For PMs:** Pinpoints "Confusion" clusters to identify unclear requirements.