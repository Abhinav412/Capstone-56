import os
import streamlit as st
import sys
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
import re

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Page config
st.set_page_config(
    page_title="Financial Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .status-healthy {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .status-unhealthy {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'api_keys_set' not in st.session_state:
    st.session_state['api_keys_set'] = False
if 'analysis_results' not in st.session_state:
    st.session_state['analysis_results'] = None

# Helper function to check FinBERT API
def check_finbert_api():
    """Check if FinBERT API is healthy"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except:
        return False, None

# Helper function to parse sentiment from crew output
def parse_sentiment_data(raw_output):
    """
    Parse sentiment analysis data from crew output.
    This is a simple parser - adjust based on your actual crew output format.
    """
    try:
        output_str = str(raw_output)
        
        # Try to extract sentiment mentions
        sentiments = {
            'positive': len(re.findall(r'\bpositive\b', output_str, re.IGNORECASE)),
            'neutral': len(re.findall(r'\bneutral\b', output_str, re.IGNORECASE)),
            'negative': len(re.findall(r'\bnegative\b', output_str, re.IGNORECASE))
        }
        
        # Calculate percentages
        total = sum(sentiments.values())
        if total > 0:
            sentiment_pct = {k: (v / total) * 100 for k, v in sentiments.items()}
        else:
            sentiment_pct = {'positive': 33.33, 'neutral': 33.33, 'negative': 33.34}
        
        return sentiment_pct, sentiments
    except:
        return {'positive': 33.33, 'neutral': 33.33, 'negative': 33.34}, {'positive': 0, 'neutral': 0, 'negative': 0}

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Status Check
    st.subheader("🔍 System Status")
    finbert_healthy, finbert_info = check_finbert_api()
    
    if finbert_healthy:
        st.markdown(f"""
        <div class="status-box status-healthy">
            ✅ <strong>FinBERT API: Connected</strong><br>
            Device: {finbert_info.get('device', 'N/A')}<br>
            Model: {finbert_info.get('model_type', 'N/A')}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-box status-unhealthy">
            ❌ <strong>FinBERT API: Not Connected</strong><br>
            Start with: <code>python sentiment_api.py</code>
        </div>
        """, unsafe_allow_html=True)
        st.warning("⚠️ FinBERT API must be running for sentiment analysis")
    
    st.markdown("---")
    
    # API Keys
    st.subheader("🔑 API Keys")
    gemini_key = st.text_input(
        "Google API Key (Gemini)",
        type="password",
        value=os.getenv("GOOGLE_API_KEY", ""),
        help="Required for Gemini LLM"
    )
    serper_key = st.text_input(
        "Serper API Key",
        type="password",
        value=os.getenv("SERPER_API_KEY", ""),
        help="Required for web search"
    )
    exa_key = st.text_input(
        "EXA API Key",
        type="password",
        value=os.getenv("EXA_API_KEY", ""),
        help="Required for deep search"
    )
    firecrawl_key = st.text_input(
        "Firecrawl API Key (Optional)",
        type="password",
        value=os.getenv("FIRECRAWL_API_KEY", ""),
        help="Optional for web scraping"
    )
    
    if st.button("💾 Save API Keys", use_container_width=True):
        if gemini_key:
            os.environ["GOOGLE_API_KEY"] = gemini_key
        if serper_key:
            os.environ["SERPER_API_KEY"] = serper_key
        if exa_key:
            os.environ["EXA_API_KEY"] = exa_key
        if firecrawl_key:
            os.environ["FIRECRAWL_API_KEY"] = firecrawl_key
        
        st.session_state['api_keys_set'] = True
        st.success("✅ API Keys saved!")
        st.rerun()
    
    # Show API key status
    if st.session_state['api_keys_set']:
        st.success("✅ API Keys configured")
    else:
        st.info("👆 Please configure API keys")
    
    st.markdown("---")
    
    # Time period
    st.subheader("📅 Analysis Period")
    time_period = st.selectbox(
        "Time Range",
        ["Last 24 Hours", "Last 3 Days", "Last Week", "Last 2 Weeks", "Last Month", "Custom"],
        index=2
    )
    
    if time_period == "Custom":
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=7))
        end_date = st.date_input("End Date", datetime.now())
    else:
        period_mapping = {
            "Last 24 Hours": 1,
            "Last 3 Days": 3,
            "Last Week": 7,
            "Last 2 Weeks": 14,
            "Last Month": 30
        }
        days = period_mapping.get(time_period, 7)
        start_date = datetime.now() - timedelta(days=days)
        end_date = datetime.now()
    
    st.info(f"📊 {(end_date - start_date).days} days selected")
    
    st.markdown("---")
    
    # About
    with st.expander("ℹ️ About This Dashboard"):
        st.markdown("""
        ### 🤖 AI-Powered Analysis
        
        **Components:**
        - **CrewAI Agents**: Multi-agent orchestration
        - **FinBERT**: Financial sentiment analysis
        - **Google Gemini**: Advanced reasoning
        
        **Workflow:**
        1. Monitor trending topics
        2. Search relevant news
        3. Scrape article content
        4. Analyze sentiment with FinBERT
        5. Generate comprehensive report
        
        **Requirements:**
        - FinBERT API running (port 8000)
        - Valid API keys configured
        """)

# Main content
st.markdown('<div class="main-header">📊 Financial Intelligence Dashboard</div>', unsafe_allow_html=True)
st.markdown("<div style='text-align: center; color: #666;'>AI-Powered Market Sentiment Analysis</div>", unsafe_allow_html=True)
st.markdown("---")

# Check prerequisites
can_run = st.session_state['api_keys_set'] and finbert_healthy

if not can_run:
    col1, col2 = st.columns(2)
    with col1:
        if not st.session_state['api_keys_set']:
            st.error("❌ API keys not configured")
    with col2:
        if not finbert_healthy:
            st.error("❌ FinBERT API not running")

# Input section
col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_input(
        "🔍 Enter topic to analyze:",
        value="Adani Group IPO",
        placeholder="e.g., 'Adani IPO', 'RBI interest rates', 'Tesla stock'",
        disabled=not can_run
    )
with col2:
    st.write("")  # Spacing
    st.write("")  # Spacing
    analyze_button = st.button(
        "🚀 Run Analysis",
        type="primary",
        use_container_width=True,
        disabled=not can_run
    )

# Clear button
if st.session_state.get('analysis_results'):
    if st.button("🗑️ Clear Results"):
        st.session_state['analysis_results'] = None
        st.rerun()

# Run analysis
if analyze_button:
    if not query:
        st.error("⚠️ Please enter a topic to analyze")
    else:
        try:
            # Import crew
            from daily_market_intelligence_monitor.crew import DailyMarketIntelligenceMonitorCrew
            
            # Progress container
            progress_container = st.container()
            
            with progress_container:
                st.info("🤖 Initializing CrewAI agents...")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Initialize
                status_text.text("Loading crew configuration...")
                progress_bar.progress(20)
                
                monitoring_source = f"Google Trends, Serper News, and EXA Search from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
                
                crew_instance = DailyMarketIntelligenceMonitorCrew()
                
                status_text.text("Starting agent execution...")
                progress_bar.progress(40)
                
                # Run crew
                with st.expander("📋 Agent Execution Log", expanded=False):
                    log_placeholder = st.empty()
                    
                    inputs = {
                        'search_topic': query,
                        'monitoring_source': monitoring_source,
                        'focus_area': query,
                    }
                    
                    result = crew_instance.crew().kickoff(inputs=inputs)
                
                progress_bar.progress(100)
                status_text.text("✅ Analysis complete!")
                
                # Store results
                st.session_state['analysis_results'] = {
                    'raw_result': result,
                    'query': query,
                    'start_date': start_date,
                    'end_date': end_date,
                    'timestamp': datetime.now()
                }
                
            st.success("🎉 Analysis completed successfully!")
            
        except ImportError as e:
            st.error(f"❌ Import Error: Could not load crew module. {str(e)}")
            st.info("💡 Make sure the crew is properly configured in `src/daily_market_intelligence_monitor/`")
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            with st.expander("🔍 Error Details"):
                st.exception(e)

# Display results
if st.session_state.get('analysis_results'):
    results = st.session_state['analysis_results']
    
    st.markdown("---")
    
    # Header
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        st.subheader(f"📊 Analysis: {results['query']}")
    with col2:
        st.caption(f"📅 Period: {results['start_date'].strftime('%Y-%m-%d')} to {results['end_date'].strftime('%Y-%m-%d')}")
    with col3:
        st.caption(f"🕐 Generated: {results['timestamp'].strftime('%H:%M:%S')}")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📝 Full Report", "📊 Sentiment Analysis", "💾 Export"])
    
    with tab1:
        st.markdown("### Detailed Analysis Report")
        
        raw_result = str(results['raw_result'])
        st.markdown(raw_result)
        
        # Parse and display key insights
        st.markdown("---")
        st.markdown("### 🔑 Key Insights")
        
        # Extract sentiment data
        sentiment_pct, sentiment_counts = parse_sentiment_data(raw_result)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Positive Sentiment",
                f"{sentiment_pct['positive']:.1f}%",
                delta=f"{sentiment_counts['positive']} mentions"
            )
        
        with col2:
            st.metric(
                "Neutral Sentiment",
                f"{sentiment_pct['neutral']:.1f}%",
                delta=f"{sentiment_counts['neutral']} mentions"
            )
        
        with col3:
            st.metric(
                "Negative Sentiment",
                f"{sentiment_pct['negative']:.1f}%",
                delta=f"{sentiment_counts['negative']} mentions"
            )
    
    with tab2:
        st.markdown("### Sentiment Distribution")
        
        sentiment_pct, _ = parse_sentiment_data(str(results['raw_result']))
        
        # Pie chart
        fig_pie = px.pie(
            values=list(sentiment_pct.values()),
            names=[k.capitalize() for k in sentiment_pct.keys()],
            color=list(sentiment_pct.keys()),
            color_discrete_map={
                'positive': '#28a745',
                'neutral': '#6c757d',
                'negative': '#dc3545'
            },
            title="Overall Sentiment Distribution"
        )
        
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Bar chart
        st.markdown("### Sentiment Breakdown")
        
        fig_bar = go.Figure(data=[
            go.Bar(
                x=['Positive', 'Neutral', 'Negative'],
                y=[sentiment_pct['positive'], sentiment_pct['neutral'], sentiment_pct['negative']],
                marker_color=['#28a745', '#6c757d', '#dc3545'],
                text=[f"{v:.1f}%" for v in sentiment_pct.values()],
                textposition='auto',
            )
        ])
        
        fig_bar.update_layout(
            title="Sentiment Analysis Results",
            yaxis_title="Percentage (%)",
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Interpretation
        st.markdown("---")
        st.markdown("### 💡 Interpretation")
        
        dominant_sentiment = max(sentiment_pct, key=sentiment_pct.get)
        
        if dominant_sentiment == 'positive':
            st.success(f"✅ **Predominantly Positive** ({sentiment_pct['positive']:.1f}%): Market sentiment appears favorable for {results['query']}")
        elif dominant_sentiment == 'negative':
            st.error(f"⚠️ **Predominantly Negative** ({sentiment_pct['negative']:.1f}%): Market sentiment shows concerns regarding {results['query']}")
        else:
            st.info(f"➡️ **Mostly Neutral** ({sentiment_pct['neutral']:.1f}%): Market sentiment is balanced for {results['query']}")
    
    with tab3:
        st.markdown("### 💾 Export Options")
        
        # Prepare export data
        raw_output = str(results['raw_result'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Text export
            st.download_button(
                label="📄 Download as Text",
                data=raw_output,
                file_name=f"analysis_{results['query'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col2:
            # JSON export
            json_data = {
                "query": results['query'],
                "start_date": results['start_date'].strftime('%Y-%m-%d'),
                "end_date": results['end_date'].strftime('%Y-%m-%d'),
                "timestamp": results['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                "analysis": raw_output,
                "sentiment": sentiment_pct
            }
            
            st.download_button(
                label="📊 Download as JSON",
                data=json.dumps(json_data, indent=2),
                file_name=f"analysis_{results['query'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        # Summary
        st.markdown("---")
        st.markdown("### 📋 Analysis Summary")
        
        summary_data = {
            "Topic": results['query'],
            "Period": f"{results['start_date'].strftime('%Y-%m-%d')} to {results['end_date'].strftime('%Y-%m-%d')}",
            "Generated": results['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
            "Word Count": len(raw_output.split()),
            "Positive %": f"{sentiment_pct['positive']:.1f}%",
            "Neutral %": f"{sentiment_pct['neutral']:.1f}%",
            "Negative %": f"{sentiment_pct['negative']:.1f}%"
        }
        
        summary_df = pd.DataFrame(list(summary_data.items()), columns=['Metric', 'Value'])
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

else:
    # Welcome message when no results
    st.info("👋 Welcome! Enter a topic and click '🚀 Run Analysis' to get started.")
    
    st.markdown("### 💡 Example Topics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **IPOs & Listings**
        - Adani IPO
        - Zomato IPO
        - Paytm listing
        """)
    
    with col2:
        st.markdown("""
        **Market Events**
        - RBI interest rates
        - Budget 2024
        - Tech sector outlook
        """)
    
    with col3:
        st.markdown("""
        **Stocks & Companies**
        - Tesla stock
        - Reliance earnings
        - TCS performance
        """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Built with ❤️ using CrewAI • FinBERT • Google Gemini<br>"
    f"Dashboard v1.0 • {datetime.now().year}"
    "</div>",
    unsafe_allow_html=True
)