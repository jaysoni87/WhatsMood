import streamlit as st
import pandas as pd
import plotly.express as px
import processor  # Importing your new engine

# --- PAGE CONFIG ---
st.set_page_config(page_title="WhatsApp Emotion AI", layout="wide", page_icon="💬")
st.title("🤖 Multilingual WhatsApp Analyzer")

# --- SESSION STATE ---
# This keeps the data alive when you change filters, so it doesn't re-process every time.
if 'df' not in st.session_state:
    st.session_state['df'] = None

# --- SIDEBAR: CONTROLS ---
st.sidebar.header("📁 Data Source")

# 1. File Uploader
uploaded_file = st.sidebar.file_uploader("Upload WhatsApp Chat (.txt)", type="txt")

# 2. Process Button
if uploaded_file is not None:
    if st.sidebar.button("🚀 Analyze Chat"):

        # Create the Progress Bar Elements
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            status_text.text("📂 Parsing & Cleaning...")

            # A. Read & Parse
            raw_text = uploaded_file.getvalue().decode("utf-8")
            df_parsed = processor.parse_whatsapp_chat(raw_text)
            st.toast(f"Parsed {len(df_parsed)} messages successfully!")

            # B. Clean & Tag Language
            df_clean = processor.process_data(df_parsed)


            # C. Define the Callback Function
            # This allows processor.py to talk back to Home.py
            def update_progress_bar(value, message):
                progress_bar.progress(value)
                status_text.text(f"🧠 {message}")


            # D. Run AI Analysis (with the callback)
            final_df = processor.analyze_emotions(df_clean, status_callback=update_progress_bar)

            # E. Finish
            progress_bar.progress(100)
            status_text.text("✅ Analysis Complete!")

            # Save to Session State & Reload
            st.session_state['df'] = final_df
            st.rerun()

        except Exception as e:
            st.error(f"Error processing file: {e}")

# --- MAIN DASHBOARD (Only shows if data exists) ---
if st.session_state['df'] is not None:
    df = st.session_state['df']

    # --- FILTERS ---
    st.sidebar.divider()
    st.sidebar.header("🔍 Filters")
    users = df['Author'].unique()
    selected_users = st.sidebar.multiselect("Select Users", users, default=users)

    # Filter Data
    df_filtered = df[df['Author'].isin(selected_users)]

    # --- AI EXECUTIVE SUMMARY SECTION ---
    st.markdown("### 🤖 AI Executive Insight")

    # 1. Try to get key from Secrets (Best Practice)
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
    else:
        # 2. Fallback: Ask user if secret is missing
        api_key = st.sidebar.text_input("🔑 Gemini API Key", type="password")

    if api_key and st.session_state['df'] is not None:
        if st.button("✨ Generate Smart Summary"):
            with st.spinner("Asking Gemini to analyze the chat..."):
                # We call the processor function
                summary = processor.generate_ai_summary(st.session_state['df'], api_key)

                # Display Result
                if "Error" in summary:
                    st.error(summary)
                else:
                    st.success("Analysis Complete")
                    st.markdown(f"**Executive Summary:**\n\n{summary}")

    elif not api_key:
        st.warning("⚠️ No API Key found. Please add it to .streamlit/secrets.toml")

    st.divider()

    # --- ROW 1: KPIs ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Messages", f"{len(df_filtered):,}")
    with col2:
        top_user = df_filtered['Author'].value_counts().idxmax() if not df_filtered.empty else "N/A"
        st.metric("Most Active", top_user)
    with col3:
        # Vibe (Most common non-neutral)
        non_neutral = df_filtered[df_filtered['Emotion_Final'] != 'neutral']
        vibe = non_neutral['Emotion_Final'].value_counts().idxmax().title() if not non_neutral.empty else "Neutral"
        st.metric("Group Vibe", vibe)
    with col4:
        confused = len(df_filtered[df_filtered['Emotion_Final'] == 'confusion'])
        st.metric("Confusion Moments", confused)

    st.divider()

    # --- ROW 2: RADAR CHART ---
    c1, c2 = st.columns([2, 1])

    with c1:
        st.subheader("🧠 Emotional Fingerprint")
        # Prepare Data
        emotion_counts = df_filtered[df_filtered['Emotion_Final'] != 'neutral'].groupby(
            ['Author', 'Emotion_Final']).size().reset_index(name='Count')

        if not emotion_counts.empty:
            fig_radar = px.line_polar(
                emotion_counts, r='Count', theta='Emotion_Final', color='Author',
                line_close=True, template="plotly_dark", title="Who feels what?"
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        else:
            st.info("Not enough emotional data to generate Radar.")

    with c2:
        st.subheader("📝 Most Used")
        # Quick Emoji Cloud
        all_text = " ".join(df_filtered['Message'].astype(str))
        # Simple extraction for display
        st.write("Top detected languages:")
        st.write(df_filtered['Lang_Tag'].value_counts().head(3))

        # --- ROW 3: TIMELINES ---
        st.subheader("📈 Emotional Volatility Timeline")

        # Filter for interesting emotions
        daily_emotions = df_filtered[
            df_filtered['Emotion_Final'].isin(['confusion', 'joy', 'anger', 'amusement', 'sadness'])]

        if not daily_emotions.empty:
            # Group by Date and Emotion
            daily_trend = daily_emotions.groupby(
                [pd.Grouper(key='Full_Time', freq='D'), 'Emotion_Final']).size().reset_index(name='Count')

            fig_line = px.line(
                daily_trend,
                x='Full_Time',
                y='Count',
                color='Emotion_Final',
                title="Emotional Trends over Time",
                template="plotly_dark"
            )
            st.plotly_chart(fig_line, use_container_width=True)

    # --- ROW 4: DATA LOG ---
    with st.expander("📂 View Analyzed Log"):
        st.dataframe(
            df_filtered[['Full_Time', 'Author', 'Message', 'Emotion_Final', 'Lang_Tag']].sort_values('Full_Time',
                                                                                                     ascending=False))

else:
    # --- LANDING PAGE ---
    st.markdown("""
    ### 👋 Welcome!
    To get started:
    1. Export your WhatsApp chat (without media).
    2. Upload the `.txt` file in the sidebar.
    3. Click **Analyze Chat**.
    """)