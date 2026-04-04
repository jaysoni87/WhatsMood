import streamlit as st
import pandas as pd
import numpy as np
import re
import hashlib
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from deep_translator import GoogleTranslator
from concurrent.futures import ThreadPoolExecutor, as_completed
import google.generativeai as genai
from functools import lru_cache
import joblib
import os
from pathlib import Path

# --- CONFIGURATION ---
CACHE_DIR = Path(".emotion_cache")
CACHE_DIR.mkdir(exist_ok=True)
CACHE_FILE = CACHE_DIR / "emotion_cache.pkl"

# Language Detection Keywords
HINGLISH_KEYWORDS = {'hai', 'nahi', 'kya', 'yaar', 'kar', 'hoon', 'tha', 'thi', 'kaise', 
                     'kyun', 'kab', 'kahan', 'mera', 'tera', 'apna', 'sab', 'sahi', 
                     'matlab', 'kuch', 'bhi', 'bahut', 'thoda', 'kaafi'}

GUJLISH_KEYWORDS = {'che', 'chhe', 'su', 'kem', 'thi', 'ne', 'pan', 'ane', 'jetla', 
                    'have', 'hatu', 'hatu', 'karu', 'karvu', 'aapde', 'tamne', 
                    'mane', 'saru', 'nathi', 'nai', 'kevi', 'kemey'}


# --- 1. SMART LANGUAGE DETECTION ---
@lru_cache(maxsize=5000)
def detect_language_type(text):
    """
    Detects if text needs translation.
    Returns: 'english', 'hinglish', 'gujlish', or 'other'
    """
    if not isinstance(text, str) or len(text) < 3:
        return 'english'
    
    text_lower = text.lower()
    words = set(re.findall(r'\b\w+\b', text_lower))
    
    # Check for native scripts (Devanagari/Gujarati)
    if re.search(r'[\u0900-\u097F\u0A80-\u0AFF]', text):
        return 'other'  # Contains native script
    
    # Check for Romanized keywords
    hinglish_matches = len(words.intersection(HINGLISH_KEYWORDS))
    gujlish_matches = len(words.intersection(GUJLISH_KEYWORDS))
    
    if hinglish_matches >= 2 or (hinglish_matches == 1 and len(words) < 5):
        return 'hinglish'
    if gujlish_matches >= 2 or (gujlish_matches == 1 and len(words) < 5):
        return 'gujlish'
    
    return 'english'


# --- 2. SMART TRANSLATION LAYER ---
class SmartTranslator:
    """Handles translation with caching to reduce API calls"""
    
    def __init__(self):
        self.translator = GoogleTranslator(source='auto', target='en')
        self.cache = {}
        self._load_cache()
    
    def _load_cache(self):
        """Load translation cache from disk"""
        cache_path = CACHE_DIR / "translation_cache.pkl"
        if cache_path.exists():
            try:
                self.cache = joblib.load(cache_path)
            except:
                self.cache = {}
    
    def _save_cache(self):
        """Save translation cache to disk"""
        cache_path = CACHE_DIR / "translation_cache.pkl"
        try:
            joblib.dump(self.cache, cache_path)
        except:
            pass
    
    def translate(self, text):
        """Translate text with caching"""
        if not isinstance(text, str) or len(text) < 2:
            return text
        
        # Create cache key
        cache_key = hashlib.md5(text.encode()).hexdigest()
        
        # Check cache
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Translate
        try:
            translated = self.translator.translate(text)
            self.cache[cache_key] = translated
            
            # Save cache periodically (every 100 new translations)
            if len(self.cache) % 100 == 0:
                self._save_cache()
            
            return translated
        except Exception as e:
            return text  # Fallback to original
    
    def __del__(self):
        """Save cache on cleanup"""
        self._save_cache()


# Global translator instance
translator = SmartTranslator()


# --- 3. FAST EMOTION MODEL ---
@st.cache_resource
def load_emotion_model():
    """
    Loads the optimized emotion model.
    Model: j-hartmann/emotion-english-distilroberta-base
    - 3x faster than RoBERTa base
    - Trained on social media text (Twitter)
    - 7 emotions: joy, sadness, anger, fear, surprise, disgust, neutral
    """
    model_name = "j-hartmann/emotion-english-distilroberta-base"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        emotion_pipeline = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            top_k=1,
            truncation=True,
            max_length=128,  # Optimal for social media
            device=-1  # CPU (use 0 for GPU if available)
        )
        
        print("✅ Fast Emotion Model Loaded Successfully")
        return emotion_pipeline
        
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        raise


# --- 4. PERSISTENT EMOTION CACHE ---
class EmotionCache:
    """Caches emotion predictions across sessions"""
    
    def __init__(self):
        self.cache = self._load_cache()
    
    def _load_cache(self):
        """Load cache from disk"""
        if CACHE_FILE.exists():
            try:
                cache = joblib.load(CACHE_FILE)
                print(f"📦 Loaded {len(cache)} cached emotions")
                return cache
            except:
                return {}
        return {}
    
    def get(self, text_hash):
        """Get cached emotion"""
        return self.cache.get(text_hash)
    
    def set(self, text_hash, emotion):
        """Set emotion in cache"""
        self.cache[text_hash] = emotion
    
    def save(self):
        """Save cache to disk"""
        try:
            joblib.dump(self.cache, CACHE_FILE)
        except:
            pass
    
    def __len__(self):
        return len(self.cache)


# Global emotion cache
emotion_cache = EmotionCache()


# --- 5. TEXT NORMALIZATION ---
def normalize_text(text):
    """
    Aggressive normalization for deduplication.
    'I loooove you!!!' → 'i love you'
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove repeated characters (goood → good)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    
    # Keep only letters, numbers, spaces, and ? for confusion detection
    text = re.sub(r'[^\w\s\?]', ' ', text)
    
    # Collapse whitespace
    text = ' '.join(text.split())
    
    return text.strip()


# --- 6. DAILY AGGREGATION ENGINE ---
def apply_daily_aggregation(df, threshold=5000):
    """
    THE SECRET SAUCE: Reduces 100k messages to ~2-5k aggregated blocks.
    Groups messages by Date + Author for efficiency.
    
    Args:
        df: Parsed DataFrame
        threshold: Message count above which to use aggregation
    
    Returns:
        Aggregated DataFrame with 'is_aggregated' flag
    """
    total_msgs = len(df)
    
    # For small chats, skip aggregation
    if total_msgs < threshold:
        df['is_aggregated'] = False
        df['original_count'] = 1
        return df
    
    print(f"🔄 Aggregating {total_msgs:,} messages...")
    
    # Group by Date + Author
    aggregated = df.groupby(['Date_Only', 'Author']).agg({
        'Message': ' '.join,  # Combine all messages
        'Full_Time': 'min',   # Keep earliest timestamp
        'Hour': 'first'
    }).reset_index()
    
    # Add metadata
    aggregated['is_aggregated'] = True
    aggregated['original_count'] = df.groupby(['Date_Only', 'Author']).size().values
    
    print(f"✅ Reduced to {len(aggregated):,} blocks ({100*(1-len(aggregated)/total_msgs):.1f}% reduction)")
    
    return aggregated


# --- 7. PARALLEL BATCH PROCESSING ---
def process_batch_parallel(texts, emotion_pipeline, max_workers=4):
    """
    Process multiple batches in parallel using ThreadPoolExecutor.
    
    Args:
        texts: List of texts to analyze
        emotion_pipeline: Loaded model
        max_workers: Number of parallel workers
    
    Returns:
        List of emotion labels
    """
    results = [None] * len(texts)
    batch_size = 64  # Optimal batch size for the model
    
    def process_single_batch(batch_info):
        """Process a single batch"""
        start_idx, batch = batch_info
        try:
            predictions = emotion_pipeline(batch)
            labels = [p[0]['label'] for p in predictions]
            return start_idx, labels
        except Exception as e:
            # Fallback to neutral
            return start_idx, ['neutral'] * len(batch)
    
    # Create batches
    batches = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batches.append((i, batch))
    
    # Process in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_batch, batch): batch for batch in batches}
        
        for future in as_completed(futures):
            start_idx, labels = future.result()
            results[start_idx:start_idx + len(labels)] = labels
    
    return results


# --- 8. MAIN EMOTION ANALYSIS ENGINE ---
def analyze_emotions(df, status_callback=None):
    """
    THE CORE ENGINE with all optimizations.
    
    Flow:
    1. Daily Aggregation (95% reduction)
    2. Language Detection
    3. Smart Translation (only when needed)
    4. Deduplication
    5. Cache Check
    6. Parallel Batch Processing
    7. Gen Z Heuristics
    """
    
    # Load model
    emotion_pipeline = load_emotion_model()
    
    # Update progress
    if status_callback:
        status_callback(0.05, "Preparing data...")
    
    # STEP 1: Daily Aggregation
    df_working = apply_daily_aggregation(df.copy())
    
    if status_callback:
        status_callback(0.10, f"Detecting & translating languages...")
    
    # STEP 2: Language Detection & Translation
    processed_texts = []
    lang_tags = []
    
    translation_count = 0
    import time as time_module
    translation_start = time_module.time()
    
    total_to_process = len(df_working)
    for idx, row in df_working.iterrows():
        text = str(row['Message'])
        
        # Detect language
        lang_type = detect_language_type(text)
        lang_tags.append(lang_type)
        
        # Translate if needed
        if lang_type in ['hinglish', 'gujlish', 'other']:
            text = translator.translate(text)
            translation_count += 1
            
            # Update progress every 100 translations
            if status_callback and translation_count % 100 == 0:
                progress = 0.10 + (idx / total_to_process * 0.10)
                status_callback(progress, f"Translating... ({translation_count} texts)")
        
        processed_texts.append(text)
    
    translation_time = time_module.time() - translation_start
    if translation_count > 0:
        print(f"🌍 Translated {translation_count:,} texts in {translation_time:.1f}s ({translation_time/translation_count*1000:.0f}ms avg)")
    
    df_working['Lang_Tag'] = lang_tags
    df_working['Processed_Text'] = processed_texts
    
    if status_callback:
        status_callback(0.20, "Normalizing text...")
    
    # STEP 3: Normalization for Deduplication
    df_working['Normalized'] = df_working['Processed_Text'].apply(normalize_text)
    
    # Remove empty
    df_working = df_working[df_working['Normalized'].str.len() > 1].copy()
    
    # STEP 4: Deduplication
    unique_texts = df_working['Normalized'].unique()
    total_unique = len(unique_texts)
    
    print(f"🧩 Deduplication: {len(df_working)} → {total_unique} unique patterns")
    
    if status_callback:
        status_callback(0.25, f"Analyzing {total_unique:,} unique patterns...")
    
    # STEP 5: Cache Check
    cache_hits = 0
    cache_misses = []
    text_to_hash = {}
    hash_to_emotion = {}
    
    for text in unique_texts:
        text_hash = hashlib.md5(text.encode()).hexdigest()
        text_to_hash[text] = text_hash
        
        # Check cache
        cached_emotion = emotion_cache.get(text_hash)
        if cached_emotion:
            hash_to_emotion[text_hash] = cached_emotion
            cache_hits += 1
        else:
            cache_misses.append(text)
    
    print(f"📦 Cache: {cache_hits} hits, {len(cache_misses)} misses")
    
    # STEP 6: Parallel Batch Processing (only for cache misses)
    if cache_misses:
        if status_callback:
            status_callback(0.30, f"Processing {len(cache_misses):,} new patterns...")
        
        # Process in parallel
        emotions = process_batch_parallel(cache_misses, emotion_pipeline, max_workers=4)
        
        # Update cache
        for text, emotion in zip(cache_misses, emotions):
            text_hash = text_to_hash[text]
            hash_to_emotion[text_hash] = emotion
            emotion_cache.set(text_hash, emotion)
    
    # Save cache
    emotion_cache.save()
    
    if status_callback:
        status_callback(0.90, "Applying heuristics...")
    
    # STEP 7: Map results back
    df_working['Emotion_Base'] = df_working['Normalized'].apply(
        lambda x: hash_to_emotion.get(text_to_hash.get(x), 'neutral')
    )
    
    # STEP 8: Gen Z Heuristics (Expanded)
    df_working['Emotion_Final'] = df_working.apply(apply_gen_z_heuristics, axis=1)
    
    if status_callback:
        status_callback(0.95, "Finalizing...")
    
    # STEP 9: Handle Aggregated Data (OPTIMIZED - was the bottleneck!)
    # If we aggregated, we need to expand back to original rows
    if df_working['is_aggregated'].any():
        # Fast vectorized approach instead of slow loops
        # Just merge the emotion/lang results back to original df
        emotion_map = df_working[['Date_Only', 'Author', 'Emotion_Final', 'Lang_Tag']].copy()
        
        # Merge efficiently
        df_final = df.merge(
            emotion_map,
            on=['Date_Only', 'Author'],
            how='left'
        )
    else:
        df_final = df.copy()
        df_final['Emotion_Final'] = df_working['Emotion_Final'].values
        df_final['Lang_Tag'] = df_working['Lang_Tag'].values
    
    # Fill any NaN
    df_final['Emotion_Final'].fillna('neutral', inplace=True)
    df_final['Lang_Tag'].fillna('english', inplace=True)
    
    if status_callback:
        status_callback(1.0, "Complete!")
    
    return df_final


# --- 9. GEN Z HEURISTICS + LOVE & CONFUSION DETECTION ---
def apply_gen_z_heuristics(row):
    """
    Minimal heuristics - only override in VERY clear cases.
    Trust the base AI model for most cases!
    
    Philosophy: Only override when we're 99% sure, not 70% sure.
    """
    message = str(row['Message']).lower()
    original_msg = str(row['Message'])
    base_emotion = row['Emotion_Base']
    
    # ═══════════════════════════════════════
    # LOVE DETECTION - Only crystal clear cases
    # ═══════════════════════════════════════
    
    # ONLY these exact phrases trigger love (very explicit)
    explicit_love_phrases = [
        'i love you', 'love you so much', 'love u so much', 'i luv you',
        'i love u', 'ily', 'love you forever', 'love you always'
    ]
    
    # Check for explicit love phrase
    if any(phrase in message for phrase in explicit_love_phrases):
        return 'love'
    
    # That's it! Everything else goes to base model.
    # Single emoji or "babe" alone is NOT enough - let AI decide!
    
    # ═══════════════════════════════════════
    # CONFUSION DETECTION - Only obvious confusion
    # ═══════════════════════════════════════
    
    # Very strong confusion signals
    strong_confusion = [
        'i dont understand', "i don't understand", 'not understanding',
        'confused', 'confusing', 'what do you mean', 'wait what',
        'samajh nahi aaya', 'samajh nahi aya'
    ]
    
    # Multiple question marks (strong signal)
    if message.count('?') >= 2:
        return 'confusion'
    
    # Explicit confusion phrase
    if any(phrase in message for phrase in strong_confusion):
        return 'confusion'
    
    # ═══════════════════════════════════════
    # GEN Z SLANG - Only the clearest cases
    # ═══════════════════════════════════════
    
    # Rule 1: Skull emoji + laughter text = joy (very clear Gen Z usage)
    if '💀' in original_msg and any(word in message for word in ['lol', 'lmao', 'haha', 'dead']):
        return 'joy'
    
    # Rule 2: Crying emoji + laughter = joy (clear sarcasm)
    if '😭' in original_msg and any(word in message for word in ['lol', 'lmao', 'haha']):
        return 'joy'
    
    # Rule 3: Single slightly-smiling emoji alone = passive aggressive
    # (but only if the message is very short)
    if original_msg.strip() == '🙂' or message.strip() == 'ok 🙂':
        return 'anger'
    
    # ═══════════════════════════════════════
    # DEFAULT: TRUST THE AI MODEL
    # ═══════════════════════════════════════
    # If none of the above clear cases, use what the AI predicted
    return base_emotion


# --- 10. PARSING & CLEANING (Unchanged) ---
def parse_whatsapp_chat(uploaded_file):
    """Parse WhatsApp chat file"""
    if hasattr(uploaded_file, 'getvalue'):
        string_data = uploaded_file.getvalue().decode("utf-8")
    elif isinstance(uploaded_file, bytes):
        string_data = uploaded_file.decode("utf-8")
    else:
        string_data = str(uploaded_file)

    lines = string_data.splitlines()

    # Patterns for start of a new message line
    pattern_android = re.compile(
        r'^(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}\s?(?:am|pm|AM|PM)?)\s-\s(.*?):\s(.*)'
    )
    pattern_ios = re.compile(
        r'^\[(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}(?::\d{2})?(?:\s?[aApP][mM])?)\]\s(.*?):\s(.*)'
    )

    data = []
    current = None

    for line in lines:
        m = pattern_android.match(line) or pattern_ios.match(line)
        if m:
            if current:
                data.append(current)
            current = list(m.groups())  # [date, time, author, message]
        elif current:
            # Continuation of previous message
            current[3] += ' ' + line.strip()

    if current:
        data.append(current)

    return data


def process_data(data):
    """Clean and structure parsed data"""
    if not data or len(data) == 0:
        raise ValueError("No valid WhatsApp messages found in the file. Please check the format.")
    
    df = pd.DataFrame(data, columns=['Date', 'Time', 'Author', 'Message'])
    
    # Ensure all columns are strings
    df['Message'] = df['Message'].fillna('').astype(str)
    df['Author'] = df['Author'].fillna('Unknown').astype(str)
    df['Date'] = df['Date'].astype(str)
    df['Time'] = df['Time'].astype(str)
    
    # Parse timestamps
    datetime_str = df['Date'] + ' ' + df['Time']

    # Try all known WhatsApp timestamp formats in order
    formats_to_try = [
        '%d/%m/%y %H:%M',  # Android 24hr short year:  18/01/24 14:30
        '%d/%m/%Y %H:%M',  # Android 24hr full year:   18/01/2024 14:30
        '%d/%m/%y %I:%M %p',  # Android 12hr short year:  18/01/24 2:30 pm
        '%d/%m/%Y %I:%M %p',  # Android 12hr full year:   18/01/2024 2:30 PM
        '%d/%m/%y %I:%M:%S %p',  # iOS short year:           18/01/24 2:30:00 pm
        '%d/%m/%Y %I:%M:%S %p',  # iOS full year:            18/01/2024 2:30:00 PM
        '%d/%m/%y %H:%M:%S',  # iOS 24hr short year:      18/01/24 14:30:00
        '%d/%m/%Y %H:%M:%S',  # iOS 24hr full year:       18/01/2024 14:30:00
        '%m/%d/%y %H:%M',  # US Android 24hr short:    01/18/24 14:30
        '%m/%d/%Y %H:%M',  # US Android 24hr full:     01/18/2024 14:30
        '%m/%d/%y %I:%M %p',  # US Android 12hr short:    01/18/24 2:30 pm
        '%m/%d/%Y %I:%M %p',  # US Android 12hr full:     01/18/2024 2:30 PM
    ]

    df['Full_Time'] = pd.NaT
    for fmt in formats_to_try:
        if df['Full_Time'].isna().any():
            parsed = pd.to_datetime(datetime_str, format=fmt, errors='coerce')
            df['Full_Time'] = df['Full_Time'].fillna(parsed)

    # Final fallback: let pandas infer the format
    if df['Full_Time'].isna().any():
        fallback = pd.to_datetime(datetime_str, infer_datetime_format=True, errors='coerce')
        df['Full_Time'] = df['Full_Time'].fillna(fallback)
    
    # Drop invalid timestamps
    df.dropna(subset=['Full_Time'], inplace=True)
    
    if len(df) == 0:
        raise ValueError("Could not parse any valid timestamps. Please check date/time format.")
    
    # Extract time features
    df['Hour'] = df['Full_Time'].dt.hour
    df['Date_Only'] = df['Full_Time'].dt.date
    df['Day_Name'] = df['Full_Time'].dt.day_name()
    df['Day_Of_Week'] = df['Full_Time'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['Is_Weekend'] = df['Day_Of_Week'].isin([5, 6])  # Saturday, Sunday
    df['Is_After_Hours'] = (df['Hour'] < 9) | (df['Hour'] >= 18)  # Before 9 AM or after 6 PM
    
    # Clean messages - ensure it's string before using .str
    def clean_message(text):
        if pd.isna(text) or text is None:
            return ""
        text = str(text)
        text = re.sub(r'<Media omitted>|This message was deleted|http\S+', '', text).strip()
        return text
    
    df['Clean_Message'] = df['Message'].apply(clean_message)
    
    # Message length
    df['Message_Length'] = df['Message'].str.len()
    
    # Remove empty messages
    df = df[df['Clean_Message'].str.len() > 1].copy()
    
    if len(df) == 0:
        raise ValueError("No valid messages after cleaning. Check if the file contains actual chat content.")
    
    return df


# NEW:
# --- 11. AI SUMMARY ---
def generate_ai_summary(df, api_key):
    """Generate executive summary using Gemini — business or personal mode."""
    if not api_key:
        return "⚠️ API Key missing."

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-flash-latest')

        # Common stats
        stats = df['Emotion_Final'].value_counts().head(5).to_dict()
        total_msgs = len(df)
        active_users = df['Author'].nunique()
        lang_stats = df['Lang_Tag'].value_counts().to_dict()
        sample = df.tail(15)[['Author', 'Message']].to_string(index=False)

        # Detect chat type from session state or fallback via participant count
        import streamlit as st
        chat_type = st.session_state.get('chat_type', 'personal')

        if chat_type == 'business':
            # --- BUSINESS PROMPT ---
            author_msg_counts = df['Author'].value_counts().head(10).to_dict()
            author_emotions = {}
            for author in df['Author'].unique():
                top_emotion = df[df['Author'] == author]['Emotion_Final'].mode()
                author_emotions[author] = top_emotion[0] if len(top_emotion) > 0 else 'neutral'

            work_life = {}
            if 'Is_After_Hours' in df.columns and 'Is_Weekend' in df.columns:
                for author in df['Author'].unique():
                    adf = df[df['Author'] == author]
                    work_life[author] = {
                        'after_hours_pct': round(adf['Is_After_Hours'].mean() * 100, 1),
                        'weekend_pct': round(adf['Is_Weekend'].mean() * 100, 1)
                    }

            prompt = f"""You are a senior organizational psychologist and team performance analyst.
You are reviewing a WhatsApp business/team chat log and must produce a sharp, data-driven executive report.

=== CHAT STATISTICS ===
- Total Messages Analyzed: {total_msgs:,}
- Team Members: {active_users}
- Top Emotions Detected: {stats}
- Languages Used: {lang_stats}

=== INDIVIDUAL CONTRIBUTION (message count per member) ===
{author_msg_counts}

=== DOMINANT EMOTION PER MEMBER ===
{author_emotions}

=== WORK-LIFE BALANCE (after-hours & weekend messaging %) ===
{work_life}

=== RECENT CONVERSATION SAMPLE (last 15 messages) ===
{sample}

=== YOUR TASK ===
Write a structured executive report with EXACTLY these 4 sections:

**1. 🏆 Top Performers**
Identify 1-2 members who are most engaged, contributing, and emotionally positive. Be specific with names and message counts.

**2. ⚠️ Disengaged or Passive Members**
Identify members with very low message counts or highly negative/neutral emotions who may be disengaged, distracted, or just passing time in the chat.

**3. 🔥 Burnout & Stress Signals**
Call out any members sending high volumes of after-hours or weekend messages. Flag anyone whose dominant emotion is anger, fear, or sadness.

**4. 💡 Manager Action Items**
Give 2-3 concrete, actionable recommendations the team manager should act on this week based on the data above.

Rules: Be direct and specific. Use member names. Do not hedge. Do not repeat statistics already shown. Keep each section to 2-3 sentences maximum."""

        else:
            # --- PERSONAL / GROUP FUN PROMPT ---
            # Per-person emotional profile
            author_emotions = {}
            for author in df['Author'].unique():
                top_3 = df[df['Author'] == author]['Emotion_Final'].value_counts().head(3).to_dict()
                author_emotions[author] = top_3

            # Most active hours
            peak_hour = df['Hour'].value_counts().idxmax() if 'Hour' in df.columns else 'N/A'
            peak_day = df['Day_Name'].value_counts().idxmax() if 'Day_Name' in df.columns else 'N/A'

            # Who initiates most (first message of each day)
            if 'Date_Only' in df.columns:
                initiator = df.sort_values('Full_Time').groupby('Date_Only').first()['Author'].value_counts().idxmax()
            else:
                initiator = 'N/A'

            # Emoji presence
            emoji_msgs = df['Message'].str.contains(
                r'[\U00010000-\U0010ffff]', regex=True, na=False
            ).sum()
            emoji_pct = round(emoji_msgs / total_msgs * 100, 1)

            prompt = f"""You are a warm and insightful relationship and communication analyst.
You are reviewing a personal WhatsApp chat and must reveal what the conversation truly says about the people involved — their moods, connection, habits, and hidden patterns.

=== CHAT OVERVIEW ===
- Total Messages: {total_msgs:,}
- People in Chat: {active_users}
- Top Emotions Overall: {stats}
- Languages Used: {lang_stats}
- Messages with Emojis: {emoji_pct}%
- Most Active Hour of Day: {peak_hour}:00
- Most Active Day of Week: {peak_day}
- Who starts the conversation most: {initiator}

=== EMOTIONAL PROFILE PER PERSON (top 3 emotions) ===
{author_emotions}

=== RECENT CONVERSATION SAMPLE ===
{sample}

=== YOUR TASK ===
Write a warm but insightful personal chat analysis with EXACTLY these 4 sections:

**1. 💬 Overall Vibe**
Describe the general mood and energy of this chat in 2-3 sentences. Is it joyful, tense, caring, playful, or something else? What does the emotional data reveal?

**2. 🧠 Personality Snapshot**
Based on emotion patterns and messaging habits, describe each person's communication personality in 1-2 sentences each. Use names. Be honest but kind.

**3. ⚡ Interesting Patterns**
Point out 2-3 genuinely interesting behavioral patterns — like who always responds first, who uses the most emojis, late-night messaging habits, emotional shifts over time, or anything surprising the data reveals.

**4. 💡 Fun Insight**
End with one memorable, feel-good (or gently humorous) observation about this chat that the people themselves would find surprising or smile-worthy.

Rules: Be conversational, warm, and human — not corporate. Use names. Avoid jargon. Make it feel like a friend who studied their chat gave them a reading."""

        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"Error generating summary: {e}"


# --- 11. CHAT TYPE DETECTION (NEW!) ---
def detect_chat_type(df):
    """
    Detect if chat is Personal/Fun or Business.
    Returns: 'personal', 'business', 'group_fun'
    """
    num_participants = df['Author'].nunique()
    total_messages = len(df)
    
    # Business keywords
    business_keywords = [
        'project', 'deadline', 'client', 'meeting', 'standup', 'sprint',
        'task', 'ticket', 'bug', 'feature', 'release', 'deploy', 'pr',
        'code review', 'testing', 'production', 'staging', 'api', 'database',
        'urgent', 'asap', 'priority', 'escalation', 'customer', 'user',
        'hr', 'manager', 'team lead', 'performance', 'review', 'evaluation',
        'quarterly', 'investor', 'demo', 'launch'
    ]
    
    # Check business keyword frequency
    business_msg_count = 0
    for keyword in business_keywords:
        business_msg_count += df['Clean_Message'].str.contains(
            keyword, case=False, na=False
        ).sum()
    
    business_keyword_ratio = business_msg_count / total_messages if total_messages > 0 else 0
    
    # Check work hours ratio (messages between 9 AM - 6 PM)
    if 'Hour' in df.columns:
        work_hours_msgs = df[(df['Hour'] >= 9) & (df['Hour'] < 18)]
        work_hours_ratio = len(work_hours_msgs) / total_messages if total_messages > 0 else 0
    else:
        work_hours_ratio = 0
    
    # Decision: Business chat indicators
    is_business = (
        (num_participants >= 8) or  # Large group (likely team)
        (business_keyword_ratio > 0.15) or  # 15%+ business keywords
        (work_hours_ratio > 0.5 and num_participants > 3)  # 50%+ during work hours with multiple people
    )
    
    if is_business:
        return 'business'
    elif num_participants == 2:
        return 'personal'
    else:
        return 'group_fun'


# --- 12. BUSINESS METRICS CALCULATION (NEW!) ---
def calculate_business_metrics(df):
    """
    Calculate business-specific metrics for team chats.
    Returns dict with metrics.
    """
    from datetime import timedelta
    
    metrics = {}
    
    # Response time analysis
    df_sorted = df.sort_values('Full_Time').copy()
    df_sorted['Time_Diff'] = df_sorted['Full_Time'].diff()
    df_sorted['Time_Diff_Minutes'] = df_sorted['Time_Diff'].dt.total_seconds() / 60
    
    # Filter reasonable response times (< 6 hours = 360 minutes)
    reasonable_responses = df_sorted[
        (df_sorted['Time_Diff_Minutes'] > 0) & 
        (df_sorted['Time_Diff_Minutes'] < 360)
    ].copy()
    
    if len(reasonable_responses) > 0:
        metrics['avg_response_time'] = reasonable_responses['Time_Diff_Minutes'].mean()
        metrics['median_response_time'] = reasonable_responses['Time_Diff_Minutes'].median()
        
        # Response time by author
        response_by_author = reasonable_responses.groupby('Author')['Time_Diff_Minutes'].agg(['mean', 'median', 'count'])
        metrics['response_by_author'] = response_by_author.to_dict('index')
    else:
        metrics['avg_response_time'] = 0
        metrics['median_response_time'] = 0
        metrics['response_by_author'] = {}
    
    # Work-life balance metrics
    if 'Is_After_Hours' in df.columns:
        metrics['after_hours_ratio'] = df['Is_After_Hours'].mean()
    else:
        # Calculate on the fly
        after_hours = ((df['Hour'] < 9) | (df['Hour'] >= 18)).mean()
        metrics['after_hours_ratio'] = after_hours
    
    if 'Is_Weekend' in df.columns:
        metrics['weekend_ratio'] = df['Is_Weekend'].mean()
    else:
        # Calculate on the fly
        weekend = df['Full_Time'].dt.dayofweek.isin([5, 6]).mean()
        metrics['weekend_ratio'] = weekend
    
    # Participation metrics
    messages_per_author = df.groupby('Author').size().to_dict()
    metrics['messages_per_author'] = messages_per_author
    
    # Active days per author
    active_days = df.groupby('Author')['Date_Only'].nunique().to_dict()
    metrics['active_days_per_author'] = active_days
    
    # Decision extraction (simple keyword matching)
    decision_keywords = [
        'we decided', 'let\'s do', 'final decision', 'agreed to', 'decision is',
        'will do', 'I\'ll handle', 'i will', 'by tomorrow', 'by friday', 'by monday',
        'deadline', 'due date', 'action item', 'todo', 'to do'
    ]
    
    decisions = []
    for idx, row in df.iterrows():
        msg_lower = str(row['Message']).lower()
        for keyword in decision_keywords:
            if keyword in msg_lower:
                decisions.append({
                    'date': row['Full_Time'],
                    'author': row['Author'],
                    'message': row['Message'][:100],  # Truncate long messages
                    'keyword': keyword
                })
                break  # Only count once per message
    
    metrics['decisions'] = pd.DataFrame(decisions) if decisions else pd.DataFrame()
    
    # Burnout risk scoring per author
    burnout_scores = {}
    for author in df['Author'].unique():
        author_df = df[df['Author'] == author].copy()
        
        if len(author_df) < 10:  # Not enough data
            burnout_scores[author] = 0
            continue
        
        # Calculate components
        if 'Is_After_Hours' in author_df.columns:
            after_hours_msgs = author_df['Is_After_Hours'].mean()
        else:
            after_hours_msgs = ((author_df['Hour'] < 9) | (author_df['Hour'] >= 18)).mean()
        
        if 'Is_Weekend' in author_df.columns:
            weekend_msgs = author_df['Is_Weekend'].mean()
        else:
            weekend_msgs = author_df['Full_Time'].dt.dayofweek.isin([5, 6]).mean()
        
        # Message length increase (overwork indicator)
        halfway = len(author_df) // 2
        if halfway > 5:
            early_length = author_df.iloc[:halfway]['Message_Length'].mean() if 'Message_Length' in author_df.columns else 0
            late_length = author_df.iloc[halfway:]['Message_Length'].mean() if 'Message_Length' in author_df.columns else 0
            length_increase = (late_length - early_length) / early_length if early_length > 0 else 0
        else:
            length_increase = 0
        
        # Participation decrease
        total_days = (df['Full_Time'].max() - df['Full_Time'].min()).days
        if total_days > 30:
            recent_cutoff = df['Full_Time'].max() - timedelta(days=14)
            recent_msgs = len(author_df[author_df['Full_Time'] >= recent_cutoff])
            earlier_msgs = len(author_df[author_df['Full_Time'] < recent_cutoff])
            
            recent_rate = recent_msgs / 14
            earlier_rate = earlier_msgs / (total_days - 14) if (total_days - 14) > 0 else 0
            
            participation_decrease = (earlier_rate - recent_rate) / earlier_rate if earlier_rate > 0 else 0
        else:
            participation_decrease = 0
        
        # Combined burnout score (0-100)
        burnout_score = (
            after_hours_msgs * 35 +  # 35% weight
            weekend_msgs * 25 +  # 25% weight
            min(max(length_increase, 0), 1) * 20 +  # 20% weight (capped at 1)
            min(max(participation_decrease, 0), 1) * 20  # 20% weight (capped at 1)
        )
        
        burnout_scores[author] = min(max(burnout_score, 0), 100)
    
    metrics['burnout_scores'] = burnout_scores
    
    # Team interaction network (who talks after whom)
    interactions = {}
    df_sorted_network = df.sort_values('Full_Time')
    for i in range(1, len(df_sorted_network)):
        prev_author = df_sorted_network.iloc[i-1]['Author']
        curr_author = df_sorted_network.iloc[i]['Author']
        
        if prev_author != curr_author:  # Don't count self-replies
            key = (prev_author, curr_author)
            interactions[key] = interactions.get(key, 0) + 1
    
    metrics['interactions'] = interactions
    
    return metrics
