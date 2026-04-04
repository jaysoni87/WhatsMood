import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_page_config(page_title="Business Intelligence", layout="wide", page_icon="💼")

# Navigation
def show_navigation():
    st.markdown("---")
    st.markdown("### 📊 Navigate:")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.page_link("Home.py", label="🏠 Home", use_container_width=True)
    with col2:
        st.page_link("pages/1_Deep_Dive.py", label="📊 Deep Dive", use_container_width=True)
    with col3:
        st.page_link("pages/2_Business_Intelligence.py", label="💼 Business Intel", use_container_width=True)
    with col4:
        st.page_link("pages/3_Team_Health.py", label="🏥 Team Health", use_container_width=True)
    st.markdown("---")

show_navigation()

# Check if data exists
if 'df' not in st.session_state or st.session_state['df'] is None:
    st.warning("⚠️ No data found! Please analyze a chat on the Home page first.")
    st.stop()

if st.session_state.get('chat_type') != 'business':
    st.info(f"ℹ️ This page is designed for business chats. Your chat was detected as: **{st.session_state.get('chat_type') or ('unknown').replace('_', ' ').title()}**")
    st.info("You can still view general analytics on the Deep Dive page!")
    st.stop()

df = st.session_state['df']
business_metrics = st.session_state.get('business_metrics', {})

st.title("💼 Business Intelligence Dashboard")
st.markdown(f"**Analyzing {len(df):,} messages from {df['Author'].nunique()} team members**")

# Overview Metrics
st.subheader("📊 Overview")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Messages", f"{len(df):,}")
with col2:
    avg_resp = business_metrics.get('avg_response_time', 0)
    st.metric("Avg Response Time", f"{avg_resp:.0f} min")
with col3:
    active_authors = df['Author'].nunique()
    st.metric("Active Team Members", active_authors)
with col4:
    decisions_count = len(business_metrics.get('decisions', pd.DataFrame()))
    st.metric("Decisions Tracked", decisions_count)

st.divider()

# Response Time Analysis
st.subheader("⏱️ Response Time Analysis")
col1, col2 = st.columns(2)

with col1:
    # Response time by author
    response_data = business_metrics.get('response_by_author', {})
    if response_data:
        resp_df = pd.DataFrame([
            {'Author': author, 'Avg Response (min)': data['mean'], 'Count': data['count']}
            for author, data in response_data.items()
            if data['count'] > 5  # Only show authors with enough data
        ]).sort_values('Avg Response (min)')
        
        if len(resp_df) > 0:
            fig = px.bar(resp_df, x='Avg Response (min)', y='Author', orientation='h',
                         title="Average Response Time by Team Member",
                         color='Avg Response (min)', color_continuous_scale='RdYlGn_r')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data for response time analysis")
    else:
        st.info("Not enough data for response time analysis")

with col2:
    # Response time distribution
    df_sorted = df.sort_values('Full_Time').copy()
    df_sorted['Time_Diff'] = df_sorted['Full_Time'].diff()
    df_sorted['Time_Diff_Minutes'] = df_sorted['Time_Diff'].dt.total_seconds() / 60
    reasonable = df_sorted[(df_sorted['Time_Diff_Minutes'] > 0) & (df_sorted['Time_Diff_Minutes'] < 360)]
    
    if len(reasonable) > 0:
        fig = px.histogram(reasonable, x='Time_Diff_Minutes', nbins=30,
                          title="Response Time Distribution (0-6 hours)",
                          labels={'Time_Diff_Minutes': 'Minutes Between Messages'},
                          color_discrete_sequence=['#667eea'])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data for distribution")

st.divider()

# Decision Tracker
st.subheader("🎯 Decision Tracker")
decisions_df = business_metrics.get('decisions', pd.DataFrame())

if isinstance(decisions_df, pd.DataFrame) and not decisions_df.empty and len(decisions_df) > 0:
    # Timeline
    decisions_by_date = decisions_df.copy()
    decisions_by_date['date'] = pd.to_datetime(decisions_by_date['date'])
    decisions_by_date['date_only'] = decisions_by_date['date'].dt.date
    
    daily_decisions = decisions_by_date.groupby('date_only').size().reset_index(name='count')
    
    fig = px.line(daily_decisions, x='date_only', y='count',
                  title="Decisions Over Time", markers=True,
                  labels={'date_only': 'Date', 'count': 'Decisions Made'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Recent decisions table
    st.markdown("**📋 Recent Decisions:**")
    recent = decisions_df.sort_values('date', ascending=False).head(10)
    display_df = recent[['date', 'author', 'message']].copy()
    display_df['date'] = pd.to_datetime(display_df['date']).dt.strftime('%Y-%m-%d %H:%M')
    st.dataframe(display_df, use_container_width=True, height=300)
    
    # Download decisions
    csv = decisions_df.to_csv(index=False)
    st.download_button("📥 Download All Decisions", csv, "decisions.csv", "text/csv")
else:
    st.info("No decisions detected in this chat. Decision keywords include: 'we decided', 'let's do', 'will handle', 'by tomorrow', 'deadline', etc.")

st.divider()

# Team Participation
st.subheader("👥 Team Participation")
col1, col2 = st.columns(2)

with col1:
    # Messages per author
    msg_counts = pd.Series(business_metrics.get('messages_per_author', {})).sort_values(ascending=False)
    
    if len(msg_counts) > 0:
        fig = px.pie(values=msg_counts.values, names=msg_counts.index,
                     title="Share of Voice", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No participation data")

with col2:
    # Active days per author
    active_days = pd.Series(business_metrics.get('active_days_per_author', {})).sort_values(ascending=False)
    
    if len(active_days) > 0:
        fig = px.bar(x=active_days.values, y=active_days.index, orientation='h',
                     title="Active Days by Team Member",
                     labels={'x': 'Days Active', 'y': 'Team Member'},
                     color=active_days.values, color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No activity data")

st.divider()

# Communication Insights
st.subheader("💬 Communication Insights")

# Word cloud of business terms
custom_stopwords = {
    'media', 'omitted', 'ok', 'okay', 'yeah', 'yes', 'no', 'thanks', 'please',
    'hai', 'che', 'chhe', 'the', 'is', 'are', 'was', 'were', 'will', 'would',
    'can', 'just', 'like', 'good', 'great', 'sure', 'fine'
}

if 'Clean_Message' in df.columns:
    text_corpus = " ".join(df['Clean_Message'].astype(str))
    if len(text_corpus) > 50:
        wc = WordCloud(width=1200, height=400, background_color='white',
                       stopwords=custom_stopwords, colormap='viridis',
                       max_words=80).generate(text_corpus)
        
        fig_wc, ax = plt.subplots(figsize=(12, 4))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig_wc)
    else:
        st.info("Not enough text for word cloud")
else:
    st.info("Text data not available")

# Footer
st.divider()

