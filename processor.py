import pandas as pd
import re
import emoji
import time
from deep_translator import GoogleTranslator
from transformers import pipeline
import streamlit as st
import google.generativeai as genai
import os

# --- 1. SETUP & CACHING ---
# We cache the model so it doesn't reload every time you click a button
@st.cache_resource
def load_emotion_model():
    return pipeline("text-classification", model="SamLowe/roberta-base-go_emotions")


# --- 2. PARSING LOGIC ---
def parse_whatsapp_chat(file_content):
    """
    Parses raw text content into a DataFrame.
    Adapts to 24hr and 12hr formats automatically.
    """
    pattern = r'^(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}(?:\s?[aApP][mM])?)\s-\s'

    data = []
    message_buffer = []
    date, time, author = None, None, None

    lines = file_content.split('\n')

    for line in lines:
        line = line.strip()
        match = re.match(pattern, line)

        if match:
            if date and author:
                data.append([date, time, author, ' '.join(message_buffer)])

            message_buffer = []
            date, time = match.groups()
            body = line[match.end():]

            if ': ' in body:
                parts = body.split(': ', 1)
                author = parts[0]
                message_buffer.append(parts[1])
            else:
                author = 'System'
                message_buffer.append(body)
        else:
            if message_buffer:
                message_buffer.append(line)

    if date and author:
        data.append([date, time, author, ' '.join(message_buffer)])

    df = pd.DataFrame(data, columns=['Date', 'Time', 'Author', 'Message'])

    # Filter System Messages
    df = df[df['Author'] != 'System']
    df = df[~df['Message'].str.contains("Messages and calls are end-to-end encrypted", case=False, na=False)]

    return df


# --- 3. CLEANING & FEATURES ---
def process_data(df):
    # A. Timestamp Conversion
    df['Full_Time'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True, errors='coerce')
    df = df.sort_values('Full_Time')

    # B. Text Cleaning Function
    def clean_text(text):
        if not isinstance(text, str): return ""
        text = re.sub(r'http\S+|www\.\S+', '', text)  # No URLs
        text = emoji.replace_emoji(text, replace='')  # No Emojis in text
        return text.lower().strip()

    # C. Language Detection (Smart Heuristic)
    def detect_language(text):
        if not isinstance(text, str) or len(text) < 2: return 'eng_Latn'

        # Native Script
        if any('\u0A80' <= c <= '\u0AFF' for c in text): return 'guj_Gujr'
        if any('\u0900' <= c <= '\u097F' for c in text): return 'hin_Deva'

        # Keyword Logic
        text_lower = text.lower()
        guj_strong = {'che', 'su', 'kem', 'niche', 'javano', 'aau', 'bhai'}
        hin_strong = {'hai', 'kya', 'kab', 'nahi', 'samjha', 'raha', 'tha'}

        words = set(text_lower.split())
        if sum(1 for w in words if w in guj_strong) > sum(1 for w in words if w in hin_strong):
            return 'guj_Latn'
        elif sum(1 for w in words if w in hin_strong) > sum(1 for w in words if w in guj_strong):
            return 'hin_Latn'
        return 'eng_Latn'

    # Apply Cleaning & LID
    df['Clean_Message'] = df['Message'].apply(clean_text)
    df = df[df['Clean_Message'] != '']  # Remove empty
    df['Lang_Tag'] = df['Clean_Message'].apply(detect_language)

    return df


# --- 4. SMART ROUTING ENGINE (Parallel + Caching) ---

# Global Cache to store translations across different runs
# (Resets when you restart the app, which is fine)
TRANSLATION_CACHE = {}


# --- 4. ZERO-LATENCY ENGINE (Batching + Local Keywords) ---
# --- 4. HIGH-PERFORMANCE AI ENGINE (BATCHED + WINDOWED) ---
# --- 4. CONFIDENCE-BASED ROUTING ENGINE (The Cascade Model) ---

# Global Cache to prevent re-translating same failures
TRANSLATION_CACHE = {}

# Native Keyword Watchlist (Forces Path B even if model is confident)
NATIVE_TRIGGERS = ['kya', 'su', 'kem', 'shu', 'kai', 'nai', 'nathi', 'ha', 'na', 'che', '???']


# # --- 6. DE-DUPLICATION ENGINE (Smart Caching for Massive Chats) ---
# def analyze_emotions(df, status_callback=None):
#     """
#     DE-DUPLICATION ENGINE:
#     1. Burst Grouping: Compress conversation into bursts.
#     2. De-Duplication: Identify UNIQUE phrases (e.g., 'Love you' appears 500 times, analyze it once).
#     3. Caching: Run AI only on unique phrases.
#     4. Mapping: Apply results to all duplicates.
#
#     Result: Massive speedup for large, repetitive chats (100k+).
#     """
#     emotion_pipeline = load_emotion_model()
#     print("🚀 Starting De-Duplication Analysis...")
#
#     # --- STEP 1: CREATE BURSTS ---
#     df = df.sort_values(by='Full_Time')
#     df['Time_Diff'] = df['Full_Time'].diff().dt.total_seconds().fillna(0)
#
#     # New Burst if: Author changes OR Time gap > 60s
#     condition = (df['Author'] != df['Author'].shift(1)) | (df['Time_Diff'] > 60)
#     df['Burst_ID'] = condition.cumsum()
#
#     # Group into Bursts
#     burst_df = df.groupby('Burst_ID')['Clean_Message'].apply(lambda x: " ".join(x.astype(str))).reset_index()
#     burst_df.rename(columns={'Clean_Message': 'Burst_Text'}, inplace=True)
#
#     print(f"⚡ Compressed into {len(burst_df)} Bursts.")
#
#     # --- STEP 2: PRE-PROCESSING (Injection) ---
#     # We clean/inject BEFORE checking for duplicates to maximize matches
#     injection_map = {
#         r'\bmaja\b': 'joy', r'\bmajama\b': 'feeling good', r'\bsaras\b': 'great',
#         r'\bbadhiya\b': 'great', r'\bmast\b': 'awesome', r'\bjalsa\b': 'party',
#         r'\bcongrats\b': 'congratulations',
#         r'\bkya\b': 'where', r'\bsu\b': 'what', r'\bkem\b': 'why',
#         r'\bshu\b': 'what', r'\bkhabar nai\b': 'confused',
#         r'\bche\b': 'is', r'\bnathi\b': 'not',
#         r'\bgando\b': 'mad', r'\bpagal\b': 'crazy', r'\bbogu\b': 'useless',
#         r'\bthik\b': 'okay', r'\bha\b': 'yes', r'\bok\b': 'okay'
#     }
#
#     burst_df['AI_Input'] = burst_df['Burst_Text'].str.lower()
#     for pattern, replacement in injection_map.items():
#         burst_df['AI_Input'] = burst_df['AI_Input'].str.replace(pattern, replacement, regex=True)
#
#     # --- STEP 3: IDENTIFY UNIQUES (The Magic Trick) ---
#     # We extract only the UNIQUE phrases.
#     # In a "Love Bird" chat, 15,000 bursts might become just 4,000 unique phrases.
#     unique_texts = burst_df['AI_Input'].unique().tolist()
#     total_uniques = len(unique_texts)
#
#     print(f"🧩 Optimization: {len(burst_df)} Total Bursts -> {total_uniques} Unique Phrases to Analyze.")
#
#     # --- STEP 4: BATCH INFERENCE (On Uniques Only) ---
#     unique_results = {}  # Map: Text -> Emotion
#     BATCH_SIZE = 64
#
#     for i in range(0, total_uniques, BATCH_SIZE):
#         batch = unique_texts[i: i + BATCH_SIZE]
#
#         # UI Update (Shows progress on Unique Phrases)
#         if status_callback and i % 5 == 0:
#             progress = i / total_uniques
#             status_callback(progress, f"Analyzing {i}/{total_uniques} unique phrases...")
#
#         truncated_batch = [t[:128] for t in batch]
#
#         try:
#             preds = emotion_pipeline(truncated_batch)
#             labels = [p['label'] for p in preds]
#
#             # Store results in dictionary
#             for text, label in zip(batch, labels):
#                 unique_results[text] = label
#         except:
#             for text in batch:
#                 unique_results[text] = 'neutral'
#
#     # --- STEP 5: MAP BACK (Spread results to duplicates) ---
#     # 1. Map Unique Result -> All Bursts
#     burst_df['Burst_Emotion'] = burst_df['AI_Input'].map(unique_results)
#
#     # 2. Map Burst Result -> All Original Messages
#     burst_map = dict(zip(burst_df['Burst_ID'], burst_df['Burst_Emotion']))
#     df['Emotion_Label'] = df['Burst_ID'].map(burst_map)
#
#     # --- STEP 6: FINAL HEURISTICS ---
#     df['Emotion_Final'] = df['Emotion_Label']
#
#     # Standard overrides (Confusion, Sarcasm, Gen Z)
#     mask_conf = (df['Message'].str.contains('\?')) & (df['Emotion_Final'] == 'neutral')
#     df.loc[mask_conf, 'Emotion_Final'] = 'confusion'
#
#     df.loc[df['Message'].str.contains('💀'), 'Emotion_Final'] = 'amusement'
#
#     mask_smile = (df['Message'].str.contains('🙂|🙃')) & (df['Emotion_Final'].isin(['neutral', 'approval', 'joy']))
#     df.loc[mask_smile, 'Emotion_Final'] = 'sarcasm'
#
#     mask_cry = df['Message'].str.contains('😭')
#     mask_laugh = df['Clean_Message'].str.contains('lol|lmao|haha|dead|funny', case=False)
#     df.loc[mask_cry & mask_laugh, 'Emotion_Final'] = 'amusement'
#
#     # Cleanup
#     df.drop(columns=['Burst_ID', 'Time_Diff'], inplace=True)
#
#     return df
# --- 4. HIGH-PERFORMANCE AI ENGINE (BATCHED + WINDOWED) ---
def analyze_emotions(df, status_callback=None):
    """
    OPTIMIZED ENGINE v2:
    1. Aggregates by 2-Hour Time Windows (Preserves AM/PM nuances).
    2. Uses Batch Inference (32x speedup).
    3. vectorized Mapping (Instant results).
    """
    emotion_pipeline = load_emotion_model()

    # --- A. SMARTER AGGREGATION (2-Hour Windows) ---
    print("🚀 Starting High-Performance Analysis...")

    # Create a 'Time_Block' column (Floors time to nearest 2 hours: 9:15 -> 8:00)
    df['Time_Block'] = df['Full_Time'].dt.floor('2h')

    # Group by (Time_Block, Author) and join text
    # This creates the "Units of Analysis"
    grouped = df.groupby(['Time_Block', 'Author'])['Clean_Message'].apply(
        lambda x: " ".join(x.astype(str))).reset_index()

    total_blocks = len(grouped)
    print(f"⚡ Compressed {len(df)} messages into {total_blocks} Analysis Blocks.")

    # --- B. BATCH INFERENCE PREP ---
    # We convert the dataframe rows into a pure list for the AI
    # This removes the overhead of pandas during the loop
    text_blocks = grouped['Clean_Message'].tolist()
    block_keys = list(zip(grouped['Time_Block'], grouped['Author']))  # We'll use this to map results back

    results = {}  # Dictionary: (Time_Block, Author) -> Emotion
    BATCH_SIZE = 32  # Process 32 blocks at once (Transformers love this)

    # --- C. THE FAST LOOP (No Sleep, Batch Processing) ---
    for i in range(0, total_blocks, BATCH_SIZE):
        batch_texts = text_blocks[i: i + BATCH_SIZE]
        batch_keys = block_keys[i: i + BATCH_SIZE]

        # UI Update
        if status_callback:
            progress = i / total_blocks
            status_callback(progress, f"Processing batch {i}/{total_blocks}...")

        # 1. TRANSLATION GATE (Simple Speed Optimization)
        # We only translate if the block is MOSTLY non-English.
        processed_batch = []
        for text in batch_texts:
            # Quick check: If > 70% ASCII, assume English (Skip Google API)
            if len(text) > 0 and (len(text.encode('utf-8')) - len(text)) / len(text) < 0.2:
                processed_batch.append(text[:512])  # Direct English
            else:
                try:
                    # Translate only small sample to save time/API limits
                    # We take first 200 chars - usually enough for sentiment
                    trans = GoogleTranslator(source='auto', target='en').translate(text[:200])
                    processed_batch.append(trans)
                except:
                    processed_batch.append(text[:512])  # Fallback to original

        # 2. BATCH INFERENCE (The Speedup)
        # We pass the WHOLE LIST to the pipeline. It handles parallelization.
        try:
            predictions = emotion_pipeline(processed_batch)
            # Extract labels from results
            for key, res in zip(batch_keys, predictions):
                results[key] = res['label']
        except Exception as e:
            # Fallback if batch fails (rare)
            for key in batch_keys:
                results[key] = 'neutral'

    # --- D. VECTORIZED MAPPING (Instant) ---
    # Instead of df.apply(axis=1), we use map()
    # We create a tuple index on the main df to match our results dictionary

    # Create a temporary mapping column
    df['Map_Key'] = list(zip(df['Time_Block'], df['Author']))
    df['Emotion_Label'] = df['Map_Key'].map(results).fillna('neutral')

    # Cleanup temp columns
    df.drop(columns=['Map_Key'], inplace=True)

    # --- E. HEURISTICS (Vectorized/Fast) ---
    # We apply heuristics using boolean masks instead of row-by-row loops
    df['Emotion_Final'] = df['Emotion_Label']  # Default

    # 1. Confusion (Keyword Search)
    confusion_keywords = ['kya', 'su', 'what', 'wait', 'kyu', 'kem', 'samjha nai', '???']
    # Create a Regex pattern for fast searching: "kya|su|what"
    pattern = '|'.join([re.escape(k) for k in confusion_keywords])

    # Filter: Emotion is Neutral AND Message contains confusion words
    mask_confusion = (df['Emotion_Label'] == 'neutral') & \
                     (df['Clean_Message'].str.contains(pattern, case=False, na=False))
    df.loc[mask_confusion, 'Emotion_Final'] = 'confusion'

    # 2. Gen Z / Sarcasm (Emoji Search)
    # 💀 = Amusement
    df.loc[df['Message'].str.contains('💀', na=False), 'Emotion_Final'] = 'amusement'

    # 🙂 = Sarcasm (if Positive/Neutral)
    mask_smile = (df['Message'].str.contains('🙂|🙃', na=False)) & \
                 (df['Emotion_Label'].isin(['neutral', 'approval', 'joy']))
    df.loc[mask_smile, 'Emotion_Final'] = 'sarcasm'

    # 😭 = Amusement (if 'lol', 'haha' present)
    mask_cry = df['Message'].str.contains('😭', na=False)
    mask_laugh_text = df['Clean_Message'].str.contains('lol|lmao|haha|dead|funny', case=False, na=False)
    # If Crying AND Laughing Text -> Amusement
    df.loc[mask_cry & mask_laugh_text, 'Emotion_Final'] = 'amusement'

    return df


def generate_ai_summary(df, api_key):
    """
    Sends daily statistics and a text sample to Gemini for a 'Managerial Summary'.
    """
    if not api_key:
        return "⚠️ API Key missing. Please provide a Google Gemini API Key."

    # 1. Configure Gemini
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-flash-latest')

    # 2. Prepare the Context
    # We don't send the whole chat (privacy/size). We send the Metrics.
    total_msgs = len(df)
    active_users = df['Author'].unique()
    top_emotions = df['Emotion_Final'].value_counts().head(3).to_dict()

    # Get a text sample (Last 20 messages) to give the AI "flavor"
    last_msgs = df.tail(20)[['Author', 'Message']].to_string(index=False)

    prompt = f"""
    You are an AI Analyst for a WhatsApp Chat Analytics Dashboard. 
    Analyze the following data and write a 3-sentence 'Executive Summary' for a Manager.

    DATA:
    - Total Messages Analyzed: {total_msgs}
    - Active Users: {', '.join(active_users)}
    - Dominant Emotions: {top_emotions}
    - Recent Conversation Sample:
    {last_msgs}

    TASK:
    1. Summarize the overall 'Vibe' of the conversation.
    2. Highlight any specific topic discussed in the recent sample.
    3. If there is 'Confusion' or 'Anger' in the stats, mention it as a risk. Otherwise, mention the engagement level.

    OUTPUT STYLE: Professional, concise, and insightful.
    """

    # 3. Call AI
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ AI Error: {e}"