import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import emoji
import numpy as np
from collections import Counter
from datetime import timedelta

# --- PAGE SETUP ---
st.set_page_config(page_title="Deep Dive Analytics", layout="wide", page_icon="📊")

# Custom CSS
st.markdown("""
<style>
    .insight-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --- NAVIGATION ---
def show_navigation():
    st.markdown("---")
    st.markdown("### 📊 Navigate:")
    chat_type = st.session_state.get('chat_type', None)
    if chat_type == 'business':
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.page_link("Home.py", label="🏠 Home", use_container_width=True)
        with col2:
            st.page_link("pages/1_Deep_Dive.py", label="📊 Deep Dive", use_container_width=True)
        with col3:
            st.page_link("pages/2_Business_Intelligence.py", label="💼 Business Intel", use_container_width=True)
        with col4:
            st.page_link("pages/3_Team_Health.py", label="🏥 Team Health", use_container_width=True)
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.page_link("Home.py", label="🏠 Home", use_container_width=True)
        with col2:
            st.page_link("pages/1_Deep_Dive.py", label="📊 Deep Dive", use_container_width=True)
    st.markdown("---")

show_navigation()

# --- LOAD DATA ---
if 'df' not in st.session_state or st.session_state['df'] is None:
    st.warning("⚠️ No data found! Please go to the **Home** page and upload a chat file first.")
    st.stop()

df = st.session_state['df']

# --- SIDEBAR ---
st.sidebar.header("👤 Focus Mode")
users = list(df['Author'].unique())
options = ["All Group"] + users
selected_user = st.sidebar.radio("Analyze for:", options)

# Date range
date_range = st.sidebar.date_input(
    "Date Range",
    value=(df['Full_Time'].min().date(), df['Full_Time'].max().date()),
    min_value=df['Full_Time'].min().date(),
    max_value=df['Full_Time'].max().date()
)

# Filter data
if selected_user == "All Group":
    df_filtered = df
    st.title("📊 Deep Dive: Entire Group")
else:
    df_filtered = df[df['Author'] == selected_user]
    st.title(f"👤 Deep Dive: {selected_user}")

df_filtered = df_filtered[
    (df_filtered['Full_Time'].dt.date >= date_range[0]) &
    (df_filtered['Full_Time'].dt.date <= date_range[1])
    ]

st.markdown(f"**Analyzing {len(df_filtered):,} messages from {date_range[0]} to {date_range[1]}**")
st.divider()

# --- QUICK STATS ---
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total Messages", f"{len(df_filtered):,}")
with col2:
    active_days = df_filtered['Date_Only'].nunique()
    st.metric("Active Days", active_days)
with col3:
    avg_per_day = len(df_filtered) / active_days if active_days > 0 else 0
    st.metric("Avg/Day", f"{avg_per_day:.0f}")
with col4:
    unique_emojis = len(set(''.join([c for c in ''.join(df_filtered['Message'].astype(str)) if c in emoji.EMOJI_DATA])))
    st.metric("Unique Emojis", unique_emojis)
with col5:
    langs = df_filtered['Lang_Tag'].nunique() if 'Lang_Tag' in df_filtered.columns else 1
    st.metric("Languages", langs)

st.divider()

# ═══════════════════════════════════════════════════════════════
# VISUALIZATION 1 & 2: ACTIVITY HEATMAP + TIME DISTRIBUTION
# ═══════════════════════════════════════════════════════════════
st.subheader("📅 Activity Patterns")

col1, col2 = st.columns([2, 1])

with col1:
    # Heatmap
    df_filtered['Hour'] = df_filtered['Full_Time'].dt.hour
    df_filtered['Day'] = df_filtered['Full_Time'].dt.day_name()

    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = df_filtered.groupby(['Day', 'Hour']).size().reset_index(name='Message_Count')

    fig_heat = px.density_heatmap(
        heatmap_data,
        x='Hour',
        y='Day',
        z='Message_Count',
        nbinsx=24,
        category_orders={"Day": days_order},
        color_continuous_scale='Viridis',
        title="When Are Messages Sent? (Hour × Day)",
        labels={'Hour': 'Hour of Day', 'Message_Count': 'Messages'}
    )
    st.plotly_chart(fig_heat, use_container_width=True)

with col2:
    # Peak hours
    hourly = df_filtered.groupby('Hour').size().reset_index(name='Count')
    fig_hourly = px.bar(
        hourly,
        x='Hour',
        y='Count',
        title="Messages by Hour",
        color='Count',
        color_continuous_scale='Purples'
    )
    st.plotly_chart(fig_hourly, use_container_width=True)

st.divider()

# ═══════════════════════════════════════════════════════════════
# VISUALIZATION 3 & 4: EMOTION ANALYSIS
# ═══════════════════════════════════════════════════════════════
st.subheader("🎭 Emotional Landscape")

col1, col2 = st.columns(2)

with col1:
    # Emotion distribution
    emotion_counts = df_filtered['Emotion_Final'].value_counts().reset_index()
    emotion_counts.columns = ['Emotion', 'Count']

    # Color mapping for emotions (including new ones)
    emotion_colors = {
        'joy': '#FFD700',
        'love': '#FF69B4',
        'sadness': '#4169E1',
        'anger': '#DC143C',
        'fear': '#8B008B',
        'surprise': '#FF8C00',
        'disgust': '#556B2F',
        'confusion': '#A9A9A9',
        'neutral': '#D3D3D3'
    }

    fig_emotion = px.bar(
        emotion_counts,
        x='Emotion',
        y='Count',
        title="Emotion Distribution",
        color='Emotion',
        color_discrete_map=emotion_colors
    )
    st.plotly_chart(fig_emotion, use_container_width=True)

with col2:
    # Emotion timeline
    emotion_daily = df_filtered.groupby(
        [pd.Grouper(key='Full_Time', freq='D'), 'Emotion_Final']
    ).size().reset_index(name='Count')

    fig_emotion_time = px.area(
        emotion_daily,
        x='Full_Time',
        y='Count',
        color='Emotion_Final',
        title="Emotional Journey Over Time",
        color_discrete_map=emotion_colors
    )
    st.plotly_chart(fig_emotion_time, use_container_width=True)

st.divider()

# ═══════════════════════════════════════════════════════════════
# VISUALIZATION 5: MESSAGE LENGTH DISTRIBUTION
# ═══════════════════════════════════════════════════════════════
st.subheader("📏 Message Length Analysis")

df_filtered['Message_Length'] = df_filtered['Message'].str.len()

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    # Histogram
    fig_length = px.histogram(
        df_filtered,
        x='Message_Length',
        nbins=50,
        title="Message Length Distribution",
        labels={'Message_Length': 'Characters', 'count': 'Messages'},
        color_discrete_sequence=['#667eea']
    )
    st.plotly_chart(fig_length, use_container_width=True)

with col2:
    st.markdown("### 📊 Stats")
    avg_length = df_filtered['Message_Length'].mean()
    median_length = df_filtered['Message_Length'].median()
    max_length = df_filtered['Message_Length'].max()

    st.metric("Average", f"{avg_length:.0f} chars")
    st.metric("Median", f"{median_length:.0f} chars")
    st.metric("Longest", f"{max_length} chars")

with col3:
    st.markdown("### 📝 Categories")
    short = len(df_filtered[df_filtered['Message_Length'] < 20])
    medium = len(df_filtered[(df_filtered['Message_Length'] >= 20) & (df_filtered['Message_Length'] < 100)])
    long = len(df_filtered[df_filtered['Message_Length'] >= 100])

    st.metric("Short (<20)", f"{short:,}")
    st.metric("Medium (20-100)", f"{medium:,}")
    st.metric("Long (100+)", f"{long:,}")

st.divider()

# ═══════════════════════════════════════════════════════════════
# VISUALIZATION 6 & 7: USER PARTICIPATION
# ═══════════════════════════════════════════════════════════════
if selected_user == "All Group":
    st.subheader("👥 User Participation")

    col1, col2 = st.columns(2)

    with col1:
        # Share of voice
        user_counts = df_filtered['Author'].value_counts().reset_index()
        user_counts.columns = ['Author', 'Messages']

        fig_users = px.pie(
            user_counts,
            values='Messages',
            names='Author',
            title="Share of Voice",
            hole=0.4
        )
        st.plotly_chart(fig_users, use_container_width=True)

    with col2:
        # User activity over time
        user_daily = df_filtered.groupby(
            [pd.Grouper(key='Full_Time', freq='D'), 'Author']
        ).size().reset_index(name='Messages')

        fig_user_time = px.line(
            user_daily,
            x='Full_Time',
            y='Messages',
            color='Author',
            title="User Activity Over Time"
        )
        st.plotly_chart(fig_user_time, use_container_width=True)

    st.divider()

# ═══════════════════════════════════════════════════════════════
# VISUALIZATION 8: SMART WORD CLOUD (DISTINCTIVE WORDS)
# ═══════════════════════════════════════════════════════════════
st.subheader("☁️ Most Distinctive Words & Phrases")

# MASSIVE stopwords list - remove all generic chat noise
custom_stopwords = {
    # Generic chat words
    'media', 'omitted', 'image', 'video', 'document', 'deleted', 'message',
    'ok', 'okay', 'yes', 'no', 'yeah', 'yep', 'nope', 'sure', 'fine',

    # Hinglish stopwords
    'hai', 'che', 'chhe', 'ka', 'ke', 'ha', 'ho', 'ne', 'to', 'thi',
    'hu', 'hoon', 'tha', 'thi', 'kar', 'kya', 'kyun', 'kaise', 'kab',

    # Gujlish stopwords
    'su', 'kem', 'pan', 'ane', 'jetla', 'have', 'hatu', 'karu', 'karvu',
    'aapde', 'tamne', 'mane', 'saru', 'nathi', 'nai', 'kevi', 'kemey',

    # English stopwords (expanded)
    'the', 'is', 'are', 'was', 'were', 'will', 'would', 'could', 'should',
    'can', 'may', 'might', 'must', 'shall', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'done', 'doing',
    'a', 'an', 'and', 'or', 'but', 'if', 'then', 'than', 'so',
    'just', 'like', 'get', 'got', 'going', 'go', 'gone', 'went',
    'know', 'knew', 'known', 'think', 'thought', 'see', 'saw', 'seen',

    # Very common filler words
    'really', 'very', 'much', 'many', 'some', 'any', 'all', 'more', 'most',
    'good', 'bad', 'nice', 'well', 'also', 'too', 'now', 'want', 'need',
    'make', 'made', 'take', 'took', 'come', 'came', 'thing', 'things',

    # Time words (usually not distinctive)
    'today', 'tomorrow', 'yesterday', 'day', 'night', 'morning', 'evening',
    'time', 'hour', 'minute', 'week', 'month', 'year',

    # Common responses
    'lol', 'lmao', 'haha', 'hehe', 'wow', 'omg', 'oh', 'ah',
    'thanks', 'thank', 'please', 'sorry', 'excuse', 'welcome'
}

text_corpus = " ".join(df_filtered['Clean_Message'].astype(str))

if len(text_corpus) > 50:
    # Generate word cloud with better settings
    wc = WordCloud(
        width=1400,
        height=600,
        background_color='black',
        stopwords=custom_stopwords,
        colormap='viridis',
        max_words=100,  # Reduced to show only most important
        collocations=True,  # Enable to show phrases
        min_font_size=10,
        relative_scaling=0.5,  # Balance between frequency and rank
        normalize_plurals=True  # Treat "project" and "projects" as same
    ).generate(text_corpus)

    fig_wc, ax = plt.subplots(figsize=(14, 6))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig_wc)

    # Show top distinctive words with frequency
    col1, col2 = st.columns(2)

    with col1:
        word_freq = wc.words_
        top_words = list(word_freq.keys())[:20]
        st.markdown("**🔥 Top 20 Distinctive Words:**")
        # Display in a nice format
        words_display = ", ".join(top_words)
        st.markdown(f"_{words_display}_")

    with col2:
        # Word frequency chart
        if len(word_freq) > 0:
            freq_df = pd.DataFrame(
                list(word_freq.items())[:15],
                columns=['Word', 'Weight']
            )
            fig_freq = px.bar(
                freq_df,
                x='Weight',
                y='Word',
                orientation='h',
                title="Word Importance Score",
                color='Weight',
                color_continuous_scale='Viridis'
            )
            fig_freq.update_layout(showlegend=False, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_freq, use_container_width=True)
else:
    st.info("Not enough text data for word cloud.")

st.divider()

# ═══════════════════════════════════════════════════════════════
# VISUALIZATION 9 & 10: EMOJI ANALYSIS
# ═══════════════════════════════════════════════════════════════
st.subheader("😂 Emoji Universe")


def extract_emojis(text):
    return ''.join(c for c in str(text) if c in emoji.EMOJI_DATA)


all_emojis = []
all_emotions = []
emoji_by_time = []

for _, row in df_filtered.iterrows():
    ems = extract_emojis(row['Message'])
    if ems:
        for char in ems:
            all_emojis.append(char)
            all_emotions.append(row['Emotion_Final'])
            emoji_by_time.append({'emoji': char, 'time': row['Full_Time']})

if all_emojis:
    col1, col2 = st.columns([2, 1])

    with col1:
        # Emoji-Emotion matrix
        emoji_df = pd.DataFrame({'Emoji': all_emojis, 'Emotion': all_emotions})
        matrix_data = emoji_df.groupby(['Emoji', 'Emotion']).size().reset_index(name='Count')

        top_emojis = emoji_df['Emoji'].value_counts().head(20).index
        matrix_data = matrix_data[matrix_data['Emoji'].isin(top_emojis)]

        emotion_colors = {
            'joy': '#FFD700', 'love': '#FF69B4', 'sadness': '#4169E1',
            'anger': '#DC143C', 'fear': '#8B008B', 'surprise': '#FF8C00',
            'disgust': '#556B2F', 'confusion': '#A9A9A9', 'neutral': '#D3D3D3'
        }

        fig_matrix = px.bar(
            matrix_data,
            x='Count',
            y='Emoji',
            color='Emotion',
            orientation='h',
            title="Top 20 Emojis with Emotional Context",
            color_discrete_map=emotion_colors
        )
        st.plotly_chart(fig_matrix, use_container_width=True)

    with col2:
        st.markdown("### 🏆 Top Emojis")
        emoji_counts = pd.Series(all_emojis).value_counts().head(15)

        for idx, (em, count) in enumerate(emoji_counts.items(), 1):
            st.markdown(f"**{idx}.** {em} × {count}")

    # Emoji usage over time
    if len(emoji_by_time) > 10:
        emoji_time_df = pd.DataFrame(emoji_by_time)
        emoji_time_df['Date'] = emoji_time_df['time'].dt.date

        top_5_emojis = pd.Series(all_emojis).value_counts().head(5).index
        emoji_time_df = emoji_time_df[emoji_time_df['emoji'].isin(top_5_emojis)]

        emoji_trend = emoji_time_df.groupby(['Date', 'emoji']).size().reset_index(name='Count')

        fig_emoji_time = px.line(
            emoji_trend,
            x='Date',
            y='Count',
            color='emoji',
            title="Top 5 Emoji Usage Over Time",
            markers=True
        )
        st.plotly_chart(fig_emoji_time, use_container_width=True)
else:
    st.info("No emojis found in this selection.")

st.divider()

# ═══════════════════════════════════════════════════════════════
# VISUALIZATION 11: LANGUAGE PATTERNS
# ═══════════════════════════════════════════════════════════════
if 'Lang_Tag' in df_filtered.columns:
    st.subheader("🌍 Multilingual Insights")

    col1, col2 = st.columns(2)

    with col1:
        lang_counts = df_filtered['Lang_Tag'].value_counts().reset_index()
        lang_counts.columns = ['Language', 'Count']

        fig_lang = px.pie(
            lang_counts,
            values='Count',
            names='Language',
            title="Language Distribution",
            hole=0.4
        )
        st.plotly_chart(fig_lang, use_container_width=True)

    with col2:
        # Language over time
        lang_time = df_filtered.groupby(
            [pd.Grouper(key='Full_Time', freq='D'), 'Lang_Tag']
        ).size().reset_index(name='Count')

        fig_lang_time = px.area(
            lang_time,
            x='Full_Time',
            y='Count',
            color='Lang_Tag',
            title="Language Usage Over Time"
        )
        st.plotly_chart(fig_lang_time, use_container_width=True)

    st.divider()

# ═══════════════════════════════════════════════════════════════
# VISUALIZATION 12: DAILY PATTERNS BY DAY OF WEEK
# ═══════════════════════════════════════════════════════════════
st.subheader("📆 Weekly Patterns")

df_filtered['DayOfWeek'] = df_filtered['Full_Time'].dt.day_name()
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

col1, col2 = st.columns(2)

with col1:
    # Messages by day
    daily_counts = df_filtered.groupby('DayOfWeek').size().reset_index(name='Count')
    daily_counts['DayOfWeek'] = pd.Categorical(daily_counts['DayOfWeek'], categories=days_order, ordered=True)
    daily_counts = daily_counts.sort_values('DayOfWeek')

    fig_days = px.bar(
        daily_counts,
        x='DayOfWeek',
        y='Count',
        title="Activity by Day of Week",
        color='Count',
        color_continuous_scale='Blues'
    )
    st.plotly_chart(fig_days, use_container_width=True)

with col2:
    # Weekend vs Weekday
    df_filtered['IsWeekend'] = df_filtered['DayOfWeek'].isin(['Saturday', 'Sunday'])
    weekend_counts = df_filtered.groupby('IsWeekend').size().reset_index(name='Count')
    weekend_counts['Type'] = weekend_counts['IsWeekend'].map({True: 'Weekend', False: 'Weekday'})

    fig_weekend = px.pie(
        weekend_counts,
        values='Count',
        names='Type',
        title="Weekday vs Weekend Activity",
        color='Type',
        color_discrete_map={'Weekday': '#667eea', 'Weekend': '#764ba2'}
    )
    st.plotly_chart(fig_weekend, use_container_width=True)

st.divider()

# ═══════════════════════════════════════════════════════════════
# VISUALIZATION 13: RESPONSE TIME ANALYSIS
# ═══════════════════════════════════════════════════════════════
if selected_user == "All Group" and len(df_filtered) > 100:
    st.subheader("⏱️ Conversation Flow Analysis")

    # Calculate time between messages
    df_sorted = df_filtered.sort_values('Full_Time')
    df_sorted['Time_Diff'] = df_sorted['Full_Time'].diff()
    df_sorted['Time_Diff_Minutes'] = df_sorted['Time_Diff'].dt.total_seconds() / 60

    # Filter out very long gaps (>6 hours = new conversation)
    active_responses = df_sorted[df_sorted['Time_Diff_Minutes'] < 360]

    if len(active_responses) > 10:
        col1, col2 = st.columns(2)

        with col1:
            # Response time distribution
            fig_response = px.histogram(
                active_responses,
                x='Time_Diff_Minutes',
                nbins=50,
                title="Response Time Distribution (0-6 hours)",
                labels={'Time_Diff_Minutes': 'Minutes Between Messages'},
                color_discrete_sequence=['#764ba2']
            )
            st.plotly_chart(fig_response, use_container_width=True)

        with col2:
            # Average response time by hour
            active_responses['Hour'] = active_responses['Full_Time'].dt.hour
            hourly_response = active_responses.groupby('Hour')['Time_Diff_Minutes'].mean().reset_index()

            fig_hourly_resp = px.line(
                hourly_response,
                x='Hour',
                y='Time_Diff_Minutes',
                title="Avg Response Time by Hour",
                markers=True,
                labels={'Time_Diff_Minutes': 'Avg Minutes'},
                color_discrete_sequence=['#667eea']
            )
            st.plotly_chart(fig_hourly_resp, use_container_width=True)

    st.divider()

# ═══════════════════════════════════════════════════════════════
# VISUALIZATION 14 & 15: EMOTION TRANSITIONS & RADAR CHART
# ═══════════════════════════════════════════════════════════════
st.subheader("🔄 Behavioral Patterns")

col1, col2 = st.columns(2)

with col1:
    # Emotion transitions (what emotion follows what)
    df_sorted = df_filtered.sort_values('Full_Time')
    df_sorted['Next_Emotion'] = df_sorted['Emotion_Final'].shift(-1)

    transitions = df_sorted.groupby(['Emotion_Final', 'Next_Emotion']).size().reset_index(name='Count')
    transitions = transitions[transitions['Count'] > 5]  # Only show significant transitions

    emotion_colors = {
        'joy': '#FFD700', 'love': '#FF69B4', 'sadness': '#4169E1',
        'anger': '#DC143C', 'fear': '#8B008B', 'surprise': '#FF8C00',
        'disgust': '#556B2F', 'confusion': '#A9A9A9', 'neutral': '#D3D3D3'
    }

    if not transitions.empty:
        fig_transitions = px.bar(
            transitions,
            x='Emotion_Final',
            y='Count',
            color='Next_Emotion',
            title="Emotion Transitions (What Follows What)",
            labels={'Emotion_Final': 'Current Emotion', 'Count': 'Frequency'},
            color_discrete_map=emotion_colors
        )
        st.plotly_chart(fig_transitions, use_container_width=True)

with col2:
    # User emotional profile (if group)
    if selected_user == "All Group" and len(users) <= 10:
        user_emotions = df_filtered.groupby(['Author', 'Emotion_Final']).size().unstack(fill_value=0)

        # Create radar chart
        fig_radar = go.Figure()

        for user in user_emotions.index[:5]:  # Top 5 users
            fig_radar.add_trace(go.Scatterpolar(
                r=user_emotions.loc[user].values,
                theta=user_emotions.columns,
                fill='toself',
                name=user
            ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            showlegend=True,
            title="User Emotional Profiles (Top 5 Users)"
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    else:
        # Single user emotion breakdown
        emotion_pct = df_filtered['Emotion_Final'].value_counts(normalize=True) * 100

        fig_single = px.bar(
            x=emotion_pct.index,
            y=emotion_pct.values,
            title=f"{selected_user}'s Emotional Profile",
            labels={'x': 'Emotion', 'y': 'Percentage (%)'},
            color=emotion_pct.index,
            color_discrete_map=emotion_colors
        )
        st.plotly_chart(fig_single, use_container_width=True)

st.divider()

# ═══════════════════════════════════════════════════════════════
# BONUS VISUALIZATION: CONVERSATION STARTERS
# ═══════════════════════════════════════════════════════════════
if selected_user == "All Group":
    st.subheader("🎬 Conversation Starters")

    # Find messages after long gaps (>2 hours = new conversation)
    df_sorted = df_filtered.sort_values('Full_Time')
    df_sorted['Gap'] = df_sorted['Full_Time'].diff()
    conversation_starters = df_sorted[df_sorted['Gap'] > timedelta(hours=2)]

    if len(conversation_starters) > 10:
        starter_users = conversation_starters['Author'].value_counts().reset_index()
        starter_users.columns = ['Author', 'Conversations Started']

        col1, col2 = st.columns([2, 1])

        with col1:
            fig_starters = px.bar(
                starter_users,
                x='Author',
                y='Conversations Started',
                title="Who Starts Conversations Most?",
                color='Conversations Started',
                color_continuous_scale='Purples'
            )
            st.plotly_chart(fig_starters, use_container_width=True)

        with col2:
            st.markdown("### 📊 Insights")
            top_starter = starter_users.iloc[0]
            st.metric("Top Starter", top_starter['Author'],
                      f"{top_starter['Conversations Started']} conversations")

            avg_starters = starter_users['Conversations Started'].mean()
            st.metric("Average", f"{avg_starters:.0f} per person")

    st.divider()

# ═══════════════════════════════════════════════════════════════
# RAW DATA EXPLORER
# ═══════════════════════════════════════════════════════════════
with st.expander("📂 View Raw Data with Filters"):
    col1, col2, col3 = st.columns(3)

    with col1:
        emotion_filter = st.multiselect(
            "Filter by emotion:",
            options=df_filtered['Emotion_Final'].unique(),
            default=None
        )

    with col2:
        if selected_user == "All Group":
            user_filter = st.multiselect(
                "Filter by user:",
                options=df_filtered['Author'].unique(),
                default=None
            )
        else:
            user_filter = [selected_user]

    with col3:
        search_term = st.text_input("Search messages:", "")

    # Apply filters
    display_df = df_filtered.copy()
    if emotion_filter:
        display_df = display_df[display_df['Emotion_Final'].isin(emotion_filter)]
    if user_filter:
        display_df = display_df[display_df['Author'].isin(user_filter)]
    if search_term:
        display_df = display_df[display_df['Message'].str.contains(search_term, case=False, na=False)]

    st.dataframe(
        display_df[['Full_Time', 'Author', 'Message', 'Emotion_Final', 'Lang_Tag']].sort_values(
            'Full_Time', ascending=False
        ),
        use_container_width=True,
        height=400
    )

    st.markdown(f"**Showing {len(display_df):,} of {len(df_filtered):,} messages**")

    # Download filtered data
    csv = display_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Data",
        data=csv,
        file_name=f"whatsmood_filtered_{selected_user}.csv",
        mime="text/csv"
    )

# --- FOOTER ---
st.divider()
st.markdown(f"""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p><strong>📊 Total Visualizations on This Page: 15+</strong></p>
    <p>Analyzed {len(df_filtered):,} messages • {df_filtered['Date_Only'].nunique()} days • {df_filtered['Author'].nunique()} users</p>
    <p>Built with ❤️ using WhatsMood's Optimized Engine</p>
</div>
""", unsafe_allow_html=True)
