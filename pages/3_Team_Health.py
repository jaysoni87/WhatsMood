import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from datetime import timedelta

st.set_page_config(page_title="Team Health", layout="wide", page_icon="🏥")

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
    st.info(f"ℹ️ This page is designed for business/team chats. Your chat was detected as: **{st.session_state.get('chat_type') or ('unknown').replace('_', ' ').title()}**")
    st.info("Team health metrics work best with 5+ team members communicating regularly.")
    st.stop()

df = st.session_state['df']
business_metrics = st.session_state.get('business_metrics', {})

st.title("🏥 Team Health Dashboard")
st.markdown(f"**Monitoring wellbeing of {df['Author'].nunique()} team members**")

# Overall Health Score
st.subheader("📊 Overall Team Health")
burnout_scores = business_metrics.get('burnout_scores', {})
if burnout_scores and len(burnout_scores) > 0:
    avg_burnout = sum(burnout_scores.values()) / len(burnout_scores)
    health_score = 100 - avg_burnout  # Invert: high burnout = low health
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Team Health Score", f"{health_score:.0f}/100",
                 delta="Good" if health_score > 70 else "Needs Attention",
                 delta_color="normal" if health_score > 70 else "inverse")
    with col2:
        after_hours = business_metrics.get('after_hours_ratio', 0)
        st.metric("After-Hours Work", f"{after_hours*100:.0f}%",
                 delta="High" if after_hours > 0.3 else "Healthy",
                 delta_color="inverse" if after_hours > 0.3 else "normal")
    with col3:
        weekend = business_metrics.get('weekend_ratio', 0)
        st.metric("Weekend Work", f"{weekend*100:.0f}%",
                 delta="High" if weekend > 0.15 else "Low",
                 delta_color="inverse" if weekend > 0.15 else "normal")
    with col4:
        high_risk = sum(1 for score in burnout_scores.values() if score > 60)
        st.metric("High Risk Members", high_risk,
                 delta="Critical" if high_risk > 0 else "None",
                 delta_color="inverse" if high_risk > 0 else "normal")
else:
    st.info("Not enough data to calculate team health metrics. Need more messages and activity.")

st.divider()

# Burnout Risk by Individual
st.subheader("🔥 Burnout Risk Analysis")

if burnout_scores and len(burnout_scores) > 0:
    burnout_df = pd.DataFrame([
        {'Team Member': member, 'Burnout Risk': score}
        for member, score in burnout_scores.items()
    ]).sort_values('Burnout Risk', ascending=False)
    
    # Color code: Green (<30), Yellow (30-60), Red (>60)
    burnout_df['Risk Level'] = burnout_df['Burnout Risk'].apply(
        lambda x: 'Low' if x < 30 else ('Medium' if x < 60 else 'High')
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = px.bar(burnout_df, x='Burnout Risk', y='Team Member', orientation='h',
                     title="Burnout Risk Score by Team Member (0-100)",
                     color='Risk Level',
                     color_discrete_map={'Low': '#28a745', 'Medium': '#ffc107', 'High': '#dc3545'})
        fig.add_vline(x=30, line_dash="dash", line_color="gray", annotation_text="Low Risk")
        fig.add_vline(x=60, line_dash="dash", line_color="orange", annotation_text="High Risk")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📋 Risk Categories")
        low_risk = len([s for s in burnout_scores.values() if s < 30])
        medium_risk = len([s for s in burnout_scores.values() if 30 <= s < 60])
        high_risk = len([s for s in burnout_scores.values() if s >= 60])
        
        st.metric("🟢 Low Risk", low_risk)
        st.metric("🟡 Medium Risk", medium_risk)
        st.metric("🔴 High Risk", high_risk)
        
        if high_risk > 0:
            st.error(f"⚠️ {high_risk} team member(s) at high burnout risk. Consider intervention.")
        else:
            st.success("✅ No team members at high burnout risk!")
else:
    st.info("Burnout risk analysis requires more activity data.")

st.divider()

# Work Patterns
st.subheader("⏰ Work-Life Balance Patterns")

col1, col2 = st.columns(2)

with col1:
    # After-hours messaging by day
    if 'Is_After_Hours' in df.columns:
        df['Date'] = df['Full_Time'].dt.date
        daily_after_hours = df[df['Is_After_Hours']].groupby('Date').size().reset_index(name='Count')
        
        if len(daily_after_hours) > 0:
            fig = px.line(daily_after_hours, x='Date', y='Count',
                          title="After-Hours Messages Over Time",
                          markers=True, color_discrete_sequence=['#dc3545'])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No after-hours messages detected")
    else:
        st.info("After-hours data not available")

with col2:
    # Weekend vs Weekday
    if 'Is_Weekend' in df.columns:
        weekend_count = df['Is_Weekend'].sum()
        weekday_count = len(df) - weekend_count
        
        fig = px.pie(values=[weekday_count, weekend_count],
                     names=['Weekday', 'Weekend'],
                     title="Weekday vs Weekend Messages",
                     color_discrete_sequence=['#28a745', '#dc3545'])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Weekend data not available")

st.divider()

# Team Interaction Network
st.subheader("🕸️ Team Communication Network")

interactions = business_metrics.get('interactions', {})
if interactions and len(interactions) > 0:
    # Build network graph
    G = nx.DiGraph()
    
    # Add edges with weights
    edge_count = 0
    for (from_author, to_author), count in interactions.items():
        if count > 3:  # Only show significant interactions
            G.add_edge(from_author, to_author, weight=count)
            edge_count += 1
    
    if len(G.nodes()) > 0 and edge_count > 0:
        try:
            # Calculate layout
            pos = nx.spring_layout(G, k=2, iterations=50)
            
            # Create edge traces
            edge_traces = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_traces.append(
                    go.Scatter(x=[x0, x1, None], y=[y0, y1, None],
                              mode='lines', line=dict(width=0.5, color='#888'),
                              hoverinfo='none', showlegend=False)
                )
            
            # Create node trace
            node_x = []
            node_y = []
            node_text = []
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(node)
            
            node_trace = go.Scatter(
                x=node_x, y=node_y, 
                mode='markers+text',
                text=node_text, 
                textposition="top center",
                marker=dict(size=20, color='#667eea'),
                hoverinfo='text',
                showlegend=False
            )
            
            # Create figure
            fig = go.Figure(data=edge_traces + [node_trace],
                           layout=go.Layout(
                               title="Who Responds to Whom? (Network shows communication flow)",
                               showlegend=False,
                               hovermode='closest',
                               margin=dict(b=0,l=0,r=0,t=40),
                               xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                               yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                               height=500
                           ))
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("💡 **Insight**: Isolated nodes may indicate team members who need more engagement. Dense clusters show strong collaboration.")
        except Exception as e:
            st.error(f"Could not generate network graph: {e}")
    else:
        st.info("Not enough interaction data to build network graph. Need more back-and-forth communication.")
else:
    st.info("Interaction network requires conversation data between multiple people.")

st.divider()

# Engagement Tracking
st.subheader("📈 Engagement Trends")

# Messages over time by author
try:
    daily_msgs = df.groupby([pd.Grouper(key='Full_Time', freq='D'), 'Author']).size().reset_index(name='Messages')
    
    if len(daily_msgs) > 0:
        fig = px.line(daily_msgs, x='Full_Time', y='Messages', color='Author',
                      title="Daily Activity by Team Member")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data for engagement tracking")
except Exception as e:
    st.info(f"Could not generate engagement chart: {e}")

# Participation summary
st.markdown("### 📊 Participation Summary")
try:
    participation = df.groupby('Author').agg({
        'Message': 'count',
        'Date_Only': 'nunique',
        'Message_Length': 'mean'
    }).round(0)
    participation.columns = ['Total Messages', 'Active Days', 'Avg Message Length']
    participation = participation.sort_values('Total Messages', ascending=False)
    
    st.dataframe(participation, use_container_width=True)
except Exception as e:
    st.info(f"Could not generate participation summary: {e}")

# Footer
st.divider()

