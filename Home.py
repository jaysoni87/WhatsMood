import streamlit as st
import pandas as pd
import plotly.express as px
import processor as processor  # Using the optimized engine
import time
from processor import detect_chat_type, calculate_business_metrics

# --- PAGE CONFIG ---
st.set_page_config(page_title="WhatsMood - AI Emotion Analyzer", layout="wide", page_icon="💬")

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Page Navigation
def show_navigation():
    st.markdown("---")
    st.markdown("### 📊 Quick Navigation:")

    # Get chat type
    chat_type = st.session_state.get('chat_type', None)

    if chat_type == 'business':
        # Show all pages for business chat
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.page_link("Home.py", label="🏠 Home", use_container_width=True)
        with col2:
            st.page_link("pages/1_Deep_Dive.py", label="📊 Deep Dive", use_container_width=True)
        with col3:
            st.page_link("pages/2_Business_Intelligence.py", label="💼 Business Intel", use_container_width=True)
        with col4:
            st.page_link("pages/3_Team_Health.py", label="🏥 Team Health", use_container_width=True)
    elif chat_type in ['personal', 'group_fun']:
        # Show only personal pages
        col1, col2 = st.columns(2)
        with col1:
            st.page_link("Home.py", label="🏠 Home", use_container_width=True)
        with col2:
            st.page_link("pages/1_Deep_Dive.py", label="📊 Deep Dive", use_container_width=True)

    st.markdown("---")

# Show navigation at top (call this after the header)
if st.session_state.get('chat_type'):
    show_navigation()

st.markdown('<p class="main-header">🤖 WhatsMood: Multilingual Emotion Intelligence</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Transform your WhatsApp chats into actionable behavioral insights</p>',
            unsafe_allow_html=True)

# --- SESSION STATE ---
if 'chat_type' not in st.session_state:
    st.session_state['chat_type'] = None
if 'business_metrics' not in st.session_state:
    st.session_state['business_metrics'] = None
if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'processing_time' not in st.session_state:
    st.session_state['processing_time'] = None

# --- SIDEBAR: CONTROLS ---
st.sidebar.header("📁 Data Source")

# Info about the system
with st.sidebar.expander("ℹ️ System Info"):
    st.markdown("""
    **Features:**
    - ✅ Multilingual (English, Hinglish, Gujlish)
    - ✅ Daily Aggregation (95% faster)
    - ✅ Smart Translation Layer
    - ✅ Gen Z Slang Detection
    - ✅ Persistent Caching
    - ✅ 7 Emotion Classes

    **Performance:**
    - 1K messages: ~5-10s
    - 10K messages: ~15-30s
    - 100K messages: ~1-2min
    """)

# File Uploader
uploaded_file = st.sidebar.file_uploader("Upload WhatsApp Chat (.txt)", type="txt")

# Advanced Options
with st.sidebar.expander("⚙️ Advanced Options"):
    use_aggregation = st.checkbox("Use Daily Aggregation", value=True,
                                  help="Groups messages by date+author for 95% speedup. Recommended for chats >5K messages.")
    show_debug = st.checkbox("Show Debug Info", value=False)

# Process Button
if uploaded_file is not None:
    if st.sidebar.button("🚀 Analyze Chat", type="primary"):

        # Start timer
        start_time = time.time()

        # Create progress tracking
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            stats_text = st.empty()

        try:
            # A. Parse
            status_text.markdown("**Step 1/4:** 📂 Parsing chat file...")
            raw_text = uploaded_file.getvalue().decode("utf-8")
            data = processor.parse_whatsapp_chat(raw_text)

            if not data:
                st.error("❌ Could not parse chat file. Please ensure it's a valid WhatsApp export.")
                st.stop()

            progress_bar.progress(10)
            stats_text.info(f"✅ Found {len(data):,} messages")

            # B. Clean & Structure
            status_text.markdown("**Step 2/4:** 🧹 Cleaning data...")
            df_clean = processor.process_data(data)
            progress_bar.progress(20)
            stats_text.info(f"✅ Cleaned {len(df_clean):,} valid messages")

            # C. Emotion Analysis with callback
            status_text.markdown("**Step 3/4:** 🧠 Analyzing emotions...")


            def update_progress(value, message):
                """Callback for real-time progress updates"""
                progress_percent = int(20 + (value * 70))  # Map 0-1 to 20-90%
                progress_bar.progress(progress_percent)
                status_text.markdown(f"**Step 3/4:** 🧠 {message}")


            final_df = processor.analyze_emotions(df_clean, status_callback=update_progress)

            # D. Finalize
            progress_bar.progress(95)
            status_text.markdown("**Step 4/4:** ✨ Finalizing...")

            # Calculate processing time
            end_time = time.time()
            processing_time = end_time - start_time

            progress_bar.progress(100)
            status_text.markdown("**✅ Analysis Complete!**")
            stats_text.success(f"🎉 Processed {len(final_df):,} messages in {processing_time:.1f} seconds")

            # Save to session state
            st.session_state['df'] = final_df
            st.session_state['processing_time'] = processing_time

            # Detect chat type and calculate business metrics
            st.session_state['chat_type'] = detect_chat_type(final_df)
            chat_type = st.session_state['chat_type']

            if chat_type == 'business':
                with st.spinner("📊 Calculating business metrics..."):
                    try:
                        business_metrics = calculate_business_metrics(final_df)
                        st.session_state['business_metrics'] = business_metrics

                        st.success(
                            f"✅ **Business chat detected!** {final_df['Author'].nunique()} team members analyzed.")

                        # Show quick business metrics
                        st.markdown("### 💼 Business Insights Preview")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            avg_burnout = sum(business_metrics['burnout_scores'].values()) / len(
                                business_metrics['burnout_scores']) if business_metrics['burnout_scores'] else 0
                            st.metric("Avg Burnout Risk", f"{avg_burnout:.0f}/100")
                        with col2:
                            st.metric("After-Hours Work", f"{business_metrics['after_hours_ratio'] * 100:.0f}%")
                        with col3:
                            decisions_count = len(business_metrics['decisions'])
                            st.metric("Decisions Tracked", decisions_count)

                        st.info("💼 Check **Business Intelligence** and **Team Health** pages for detailed insights!")
                    except Exception as e:
                        st.warning(f"Could not calculate all business metrics: {e}")
            else:
                st.success(f"✅ **{chat_type.replace('_', ' ').title()} chat analyzed!**")

            # Show navigation after analysis
            show_navigation()

            # Debug info
            if show_debug:
                with st.expander("🔍 Debug Information"):
                    st.write(f"Cache size: {len(processor.emotion_cache)} emotions")
                    st.write(f"Translation cache: {len(processor.translator.cache)} translations")
                    st.write(f"Language distribution: {final_df['Lang_Tag'].value_counts().to_dict()}")

            time.sleep(1)
            st.rerun()

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
            if show_debug:
                st.exception(e)
        except ValueError as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("💡 **Tip**: Make sure your file is a valid WhatsApp chat export.")
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info(
                "💡 **Troubleshooting**: \n- Check if the file format matches WhatsApp export format\n- Try exporting the chat again")

# --- MAIN DASHBOARD ---
if st.session_state['df'] is not None:
    df = st.session_state['df']

    # Performance badge
    if st.session_state['processing_time']:
        st.sidebar.success(f"⚡ Processed in {st.session_state['processing_time']:.1f}s")

    # --- FILTERS ---
    st.sidebar.divider()
    st.sidebar.header("🔍 Filters")
    users = df['Author'].unique()
    selected_users = st.sidebar.multiselect("Select Users", users, default=users)

    # Date range filter
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(df['Full_Time'].min(), df['Full_Time'].max()),
        min_value=df['Full_Time'].min(),
        max_value=df['Full_Time'].max()
    )

    # Filter Data
    df_filtered = df[
        (df['Author'].isin(selected_users)) &
        (df['Full_Time'].dt.date >= date_range[0]) &
        (df['Full_Time'].dt.date <= date_range[1])
        ]

    # --- AI EXECUTIVE SUMMARY ---
    st.markdown("### 🤖 AI Executive Insight")

    # Get API key from secrets (prioritize secrets.toml)
    api_key = None
    using_secrets = False

    try:
        if hasattr(st, 'secrets') and st.secrets and "GEMINI_API_KEY" in st.secrets:
            api_key = st.secrets["GEMINI_API_KEY"]
            using_secrets = True
    except:
        pass  # Secrets file doesn't exist

    # ONLY show manual input if no secrets configured (completely hidden otherwise)
    if not using_secrets:
        with st.sidebar.expander("🔑 Setup API Key (Optional)"):
            st.markdown("""
            **Option 1: Use secrets.toml (Recommended)**
            Create `.streamlit/secrets.toml` in your project folder:
            ```toml
            GEMINI_API_KEY = "your-api-key-here"
            ```

            **Option 2: Enter manually below**
            """)
            api_key = st.text_input("Gemini API Key", type="password",
                                    help="Get your free API key from https://makersuite.google.com/app/apikey")
    # If using secrets, don't show anything - it's automatic!

    if api_key:
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("✨ Generate Smart Summary", use_container_width=True):
                with st.spinner("🤖 Asking Gemini to analyze the chat..."):
                    summary = processor.generate_ai_summary(df_filtered, api_key)

                    if "Error" in summary:
                        st.error(summary)
                    else:
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                    padding: 1.5rem; border-radius: 10px; color: white;">
                        <h4 style="margin:0; color: white;">📊 Executive Summary</h4>
                        <p style="margin-top: 1rem; line-height: 1.6;">{summary}</p>
                        </div>
                        """, unsafe_allow_html=True)
        with col2:
            st.metric("Messages Analyzed", f"{len(df_filtered):,}")
    else:
        st.info("💡 Add Gemini API key to generate AI summaries")

    st.divider()

    # --- KPI METRICS ---
    st.subheader("📊 Key Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Messages", f"{len(df_filtered):,}")

    with col2:
        top_user = df_filtered['Author'].value_counts().idxmax() if not df_filtered.empty else "N/A"
        st.metric("Most Active", top_user)

    with col3:
        # Dominant emotion (excluding neutral)
        non_neutral = df_filtered[df_filtered['Emotion_Final'] != 'neutral']
        if not non_neutral.empty:
            vibe = non_neutral['Emotion_Final'].value_counts().idxmax().title()
            vibe_count = non_neutral['Emotion_Final'].value_counts().max()
            st.metric("Dominant Emotion", vibe, f"{vibe_count:,} times")
        else:
            st.metric("Dominant Emotion", "Neutral")

    with col4:
        # Multilingual stats
        lang_diversity = df_filtered['Lang_Tag'].nunique()

        lang_counts = df_filtered['Lang_Tag'].value_counts()
        primary_lang = lang_counts.idxmax() if not lang_counts.empty else "N/A"

        st.metric("Languages", lang_diversity, primary_lang.title() if primary_lang != "N/A" else "N/A")

    with col5:
        # Active days
        active_days = df_filtered['Date_Only'].nunique()
        st.metric("Active Days", active_days)

    st.divider()

    # --- EMOTIONAL FINGERPRINT ---
    st.subheader("🧠 Emotional Fingerprint by User")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Prepare radar data (exclude neutral for clearer viz)
        non_neutral_df = df_filtered[df_filtered['Emotion_Final'] != 'neutral']

        if not non_neutral_df.empty:
            emotion_counts = non_neutral_df.groupby(['Author', 'Emotion_Final']).size().reset_index(name='Count')

            fig_radar = px.line_polar(
                emotion_counts,
                r='Count',
                theta='Emotion_Final',
                color='Author',
                line_close=True,
                template="plotly_dark",
                title="Emotional Profile Comparison"
            )
            fig_radar.update_traces(fill='toself', opacity=0.6)
            st.plotly_chart(fig_radar, use_container_width=True)
        else:
            st.info("Not enough emotional diversity to generate radar chart")

    with col2:
        st.markdown("#### 📈 Emotion Distribution")
        emotion_dist = df_filtered['Emotion_Final'].value_counts()

        fig_pie = px.pie(
            values=emotion_dist.values,
            names=emotion_dist.index,
            hole=0.4,
            title="Overall Emotions"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()

    # --- EMOTIONAL TRENDS ---
    st.subheader("📈 Emotional Trends Over Time")

    # BULLETPROOF emotion selection - handles all edge cases
    try:
        # Get all unique emotions in filtered data
        all_emotions = sorted(list(df_filtered['Emotion_Final'].unique()))

        if len(all_emotions) == 0:
            st.warning("No emotion data available for the selected filters.")
        else:
            # Smart defaults: pick top 3 most common (excluding neutral if possible)
            non_neutral = df_filtered[df_filtered['Emotion_Final'] != 'neutral']

            if len(non_neutral) > 0 and non_neutral['Emotion_Final'].nunique() > 0:
                # Get top 3 non-neutral emotions
                top_emotions_series = non_neutral['Emotion_Final'].value_counts().head(3)
                default_emotions = [e for e in top_emotions_series.index.tolist() if e in all_emotions]
            else:
                # Fallback: just pick first available emotion
                default_emotions = [all_emotions[0]]

            # Final safety check: ensure all defaults exist in options
            default_emotions = [e for e in default_emotions if e in all_emotions]
            if not default_emotions and all_emotions:
                default_emotions = [all_emotions[0]]

            selected_emotions = st.multiselect(
                "Select emotions to track:",
                options=all_emotions,
                default=default_emotions if default_emotions else []
            )
    except Exception as e:
        st.error(f"Error in emotion selection: {e}")

    selected_emotions = []

    if selected_emotions:
        timeline_df = df_filtered[df_filtered['Emotion_Final'].isin(selected_emotions)]
        daily_trend = timeline_df.groupby(
            [pd.Grouper(key='Full_Time', freq='D'), 'Emotion_Final']
        ).size().reset_index(name='Count')

        fig_line = px.line(
            daily_trend,
            x='Full_Time',
            y='Count',
            color='Emotion_Final',
            title="Daily Emotional Trends",
            template="plotly_dark",
            markers=True
        )
        fig_line.update_xaxes(title="Date")
        fig_line.update_yaxes(title="Message Count")
        st.plotly_chart(fig_line, use_container_width=True)

    st.divider()

    # --- LANGUAGE INSIGHTS ---
    st.subheader("🌍 Multilingual Analysis")

    col1, col2 = st.columns(2)

    with col1:
        lang_counts = df_filtered['Lang_Tag'].value_counts().reset_index()
        lang_counts.columns = ['Language', 'Count']

        fig_lang = px.bar(
            lang_counts,
            x='Language',
            y='Count',
            title="Language Distribution",
            color='Language',
            template="plotly_dark"
        )
        st.plotly_chart(fig_lang, use_container_width=True)

    with col2:
        # Emotion by language
        lang_emotion = df_filtered.groupby(['Lang_Tag', 'Emotion_Final']).size().reset_index(name='Count')

        fig_lang_emo = px.bar(
            lang_emotion,
            x='Lang_Tag',
            y='Count',
            color='Emotion_Final',
            title="Emotions by Language",
            template="plotly_dark",
            barmode='stack'
        )
        st.plotly_chart(fig_lang_emo, use_container_width=True)

    st.divider()

    # --- DATA EXPLORER ---
    with st.expander("📂 View Analyzed Data"):
        st.dataframe(
            df_filtered[['Full_Time', 'Author', 'Message', 'Emotion_Final', 'Lang_Tag']].sort_values(
                'Full_Time', ascending=False
            ),
            use_container_width=True,
            height=400
        )

        # Download button
        csv = df_filtered.to_csv(index=False)
        st.download_button(
            label="📥 Download Analysis (CSV)",
            data=csv,
            file_name="whatsmood_analysis.csv",
            mime="text/csv"
        )

else:
    # --- LANDING PAGE ---
    st.markdown("""
    <div style="text-align: center; padding: 3rem;">
        <h2>👋 Welcome to WhatsMood!</h2>
        <p style="font-size: 1.2rem; color: #666; margin: 2rem 0;">
            Transform your WhatsApp conversations into actionable insights
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: #262730; border-radius: 10px;">
            <h3>🚀 Lightning Fast</h3>
            <p>Process 100K+ messages in under 2 minutes</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: #262730; border-radius: 10px;">
            <h3>🌍 Multilingual</h3>
            <p>Supports English, Hinglish, and Gujarati</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: #262730; border-radius: 10px;">
            <h3>🧠 AI-Powered</h3>
            <p>7 emotion classes + Gen Z slang detection</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    ### 📝 How to Get Started:

    1. **Export your WhatsApp chat:**
       - Open the chat in WhatsApp
       - Tap the menu (⋮) → More → Export chat
       - Choose "Without Media"

    2. **Upload the `.txt` file** in the sidebar

    3. **Click "Analyze Chat"** and wait for the magic! ✨

    4. **Explore insights:**
       - Emotional fingerprints
       - Activity patterns
       - AI-generated summaries
       - Multilingual analysis
    """)

    st.info("💡 **Pro Tip:** For best results, export chats with at least 100+ messages!")
