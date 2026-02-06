import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter

# --- PAGE SETUP ---
st.set_page_config(page_title="Deep Dive Analytics", layout="wide", page_icon="📊")

# --- LOAD DATA FROM SESSION STATE ---
# We grab the data processed in Home.py
if 'df' not in st.session_state or st.session_state['df'] is None:
    st.warning("⚠️ No data found! Please go to the **Home** page and upload a chat file first.")
    st.stop()

df = st.session_state['df']

# --- SIDEBAR: USER SELECTION ---
st.sidebar.header("👤 Focus Mode")
# Get list of users
users = list(df['Author'].unique())
# Add an "All Group" option
options = ["All Group"] + users
selected_user = st.sidebar.radio("Analyze for:", options)

# Filter Data based on selection
if selected_user == "All Group":
    df_filtered = df
    st.title("📊 Deep Dive: Entire Group")
else:
    df_filtered = df[df['Author'] == selected_user]
    st.title(f"👤 Deep Dive: {selected_user}")

st.divider()

# --- SECTION 1: ACTIVITY HEATMAP (Visual 3) ---
st.subheader("📅 When are we active?")
c1, c2 = st.columns([2, 1])

with c1:
    # Prepare Heatmap Data
    df_filtered['Hour'] = df_filtered['Full_Time'].dt.hour
    df_filtered['Day'] = df_filtered['Full_Time'].dt.day_name()

    # Order days correctly
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
        title=f"Activity Heatmap ({selected_user})"
    )
    st.plotly_chart(fig_heat, use_container_width=True)

with c2:
    st.markdown("### 📊 Message Distribution")
    if selected_user == "All Group":
        # Pie chart of who speaks the most
        msg_counts = df['Author'].value_counts().reset_index()
        msg_counts.columns = ['Author', 'Count']
        fig_pie = px.pie(msg_counts, values='Count', names='Author', hole=0.4, title="Share of Voice")
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        # User Stats
        total_msgs = len(df_filtered)
        group_total = len(df)
        percent = (total_msgs / group_total) * 100
        st.metric(f"Messages by {selected_user}", f"{total_msgs:,}")
        st.metric("Contribution to Group", f"{percent:.1f}%")

st.divider()

# --- SECTION 2: WORD CLOUD (Visual 4) ---
# Logic: We use 'Clean_Message' (Romanized) to avoid Square Boxes for Gujarati/Hindi
st.subheader("☁️ Vocabulary & Word Cloud")

# Stopwords to remove (Common Hinglish/Gujlish noise)
custom_stopwords = {'media', 'omitted', 'image', 'video', 'document', 'hai', 'che', 'ka', 'ke', 'ok', 'ha', 'ho', 'ne',
                    'to', 'thi', 'chhe', 'hu'}

# Combine text
text_corpus = " ".join(df_filtered['Clean_Message'].astype(str))

if len(text_corpus) > 10:
    # Create WordCloud
    wc = WordCloud(
        width=800,
        height=400,
        background_color='black',
        stopwords=custom_stopwords,
        colormap='rainbow'
    ).generate(text_corpus)

    # Display using Matplotlib
    fig_wc, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig_wc)
else:
    st.info("Not enough text data for Word Cloud.")

st.divider()

# --- SECTION 3: THE EMOJI MATRIX (Visual 5) ---
st.subheader("😂 The Emoji-Emotion Matrix")

# 1. Expand the Emoji List (because one message can have multiple emojis)
# We assume processor.py creates an 'Emoji_List' or we extract it now
import emoji


def extract_emojis(text):
    return ''.join(c for c in text if c in emoji.EMOJI_DATA)


# Extract emojis on the fly if needed
all_emojis = []
all_emotions = []

# We iterate to pair every emoji with its message's emotion
for _, row in df_filtered.iterrows():
    ems = extract_emojis(row['Message'])
    if ems:
        for char in ems:
            all_emojis.append(char)
            all_emotions.append(row['Emotion_Final'])

if all_emojis:
    emoji_df = pd.DataFrame({'Emoji': all_emojis, 'Emotion': all_emotions})

    # Count occurrences
    matrix_data = emoji_df.groupby(['Emoji', 'Emotion']).size().reset_index(name='Count')

    # Filter for top 20 emojis to keep chart clean
    top_emojis = emoji_df['Emoji'].value_counts().head(15).index
    matrix_data = matrix_data[matrix_data['Emoji'].isin(top_emojis)]

    # Stacked Bar Chart
    fig_matrix = px.bar(
        matrix_data,
        x='Count',
        y='Emoji',
        color='Emotion',
        orientation='h',
        title=f"How {selected_user} uses Emojis (Sentiment Context)",
        template="plotly_dark"
    )
    st.plotly_chart(fig_matrix, use_container_width=True)
else:
    st.info("No emojis found in this selection.")