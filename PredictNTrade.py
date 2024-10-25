import streamlit as st
import eikon as ek
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go
import yfinance as yf
import pandas_ta as ta
from plotly.subplots import make_subplots
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import base64

nltk.download('vader_lexicon')

# Setting Eikon API key
ek.set_app_key('c90273df8e474db5a81d024f25bb21f59ff471fe')

# Function to fetch ESG and financial data using Eikon API
def fetch_data(ticker, start_date, end_date):
    fields = [
        'TR.TRESGScore', 'TR.EnvironmentPillarScore', 'TR.SocialPillarScore',
        'TR.GovernancePillarScore', 'TR.TotalReturnYTD', 'TR.ReturnOnEquityPercent',
        'TR.RevenueGrowthPercent', 'TR.TotalEquity'
    ]
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    data, err = ek.get_data(ticker, fields, {'SDate': start_date_str, 'EDate': end_date_str, 'Frq': 'D'})
    if err:
        st.error(f"Error fetching data: {err}")
        return None
    
    # Checking if 'Date' is in columns, otherwise set index as datetime if it's not already
    if 'Date' in data.columns:
        data.index = pd.to_datetime(data['Date'])
        data.drop(columns=['Date'], inplace=True)
    elif not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)  # Assuming the index is date string if 'Date' column isn't present

    return data.rename(columns={
        'TR.TRESGScore': 'ESG Score',
        'TR.EnvironmentPillarScore': 'Environmental Pillar Score',
        'TR.SocialPillarScore': 'Social Pillar Score',
        'TR.GovernancePillarScore': 'Governance Pillar Score',
        'TR.TotalReturnYTD': 'YTD Total Return',
        'TR.ReturnOnEquityPercent': 'Return on Equity',
        'TR.RevenueGrowthPercent': 'Revenue Growth',
        'TR.TotalEquity': 'Total Equity'
    })

# Regression to predict ESG scores
def perform_regression(data):
    scaler = StandardScaler()
    X = data[['Environmental Pillar Score', 'Social Pillar Score', 'Governance Pillar Score', 'YTD Total Return', 'Total Equity']]
    y = data['ESG Score']
    X_scaled = scaler.fit_transform(X)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    future_dates = pd.date_range(start=datetime.now(), periods=5, freq='A')
    future_data = pd.DataFrame(index=future_dates, columns=X.columns)
    future_data_filled = scaler.transform(future_data.fillna(0))
    predictions = model.predict(future_data_filled)
    avg_score = np.mean(predictions)
    st.metric(label="Average Predicted ESG Score for Next 5 Years", value=f"{avg_score:.2f}")
    return pd.DataFrame(predictions, index=future_dates, columns=['Predicted ESG Score'])

# Fetch and plot real-time technical indicators using Yahoo Finance API
def fetch_and_plot_technical_indicators(ticker):
    df = yf.download(ticker, period="1d", interval="5m")
    df.ta.bbands(length=20, append=True)
    df.ta.macd(append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.sma(close='close', length=20, append=True)

    fig = make_subplots(rows=4, cols=1, subplot_titles=("Bollinger Bands", "MACD", "RSI", "Moving Average (SMA 20)"),
                        shared_xaxes=True, vertical_spacing=0.1)

    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=df['BBL_20_2.0'], name='Lower BB', line=dict(color='red')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BBM_20_2.0'], name='Middle BB', line=dict(color='orange')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BBU_20_2.0'], name='Upper BB', line=dict(color='green')), row=1, col=1)

    # MACD
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_12_26_9'], name='MACD', line=dict(color='purple')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACDs_12_26_9'], name='Signal Line', line=dict(color='yellow')), row=2, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI_14'], name='RSI', line=dict(color='magenta')), row=3, col=1)

    # Moving Average
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', line=dict(color='blue')), row=4, col=1)

    fig.update_layout(height=800, title_text=f"Real-time Technical Analysis for {ticker}", template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)

# Function to fetch market data using Yahoo Finance API
def fetch_market_data_yf(ticker):
    ticker_data = yf.Ticker(ticker)
    data = ticker_data.info
    return {
        "Closing Price": data.get("regularMarketPreviousClose", "N/A"),
        "Market Cap": data.get("marketCap", "N/A"),
        "P/E Ratio": data.get("trailingPE", "N/A")
    }

# Function to fetch ESG-related news using Yahoo Finance
def fetch_esg_news_yahoo(company_name):
    query = f"{company_name} ESG news"
    url = f"https://news.search.yahoo.com/search?p={query}&fr2=sb-top"
    response = requests.get(url)
    if response.status_code != 200:
        st.error("Error fetching news data")
        return pd.DataFrame()

    soup = BeautifulSoup(response.content, 'html.parser')
    headlines = soup.find_all('h4', {'class': 's-title'})

    articles = []
    for headline in headlines:
        title = headline.text
        link = headline.find('a')['href']
        articles.append({'title': title, 'url': link})

    news_df = pd.DataFrame(articles)
    return news_df

# Function to save news to a file
def save_news_to_file(news_df, file_path):
    news_df.to_csv(file_path, index=False)

# Function to classify news articles based on sentiment
def classify_news(news_df):
    sid = SentimentIntensityAnalyzer()
    
    def get_sentiment(text):
        scores = sid.polarity_scores(text)
        if scores['compound'] >= 0.05:
            return 1  # Positive sentiment
        elif scores['compound'] <= -0.05:
            return 0  # Negative sentiment
        else:
            return -1  # Neutral sentiment

    news_df['sentiment'] = news_df['title'].apply(get_sentiment)
    return news_df

# Analyzing and displaying news sentiment
def analyze_and_display_news(news_df):
    classified_news = classify_news(news_df)
    good_news = classified_news[classified_news['sentiment'] == 1]
    bad_news = classified_news[classified_news['sentiment'] == 0]
    neutral_news = classified_news[classified_news['sentiment'] == -1]

    st.write("## Classified News")
    st.write("### Good News")
    st.write(good_news[['title', 'url']])
    st.write("### Bad News")
    st.write(bad_news[['title', 'url']])
    st.write("### Neutral News")
    st.write(neutral_news[['title', 'url']])

    total_news = len(classified_news)
    good_count = len(good_news)
    bad_count = len(bad_news)
    neutral_count = len(neutral_news)

    if total_news > 0:
        st.write(f"Good News: {good_count} ({good_count/total_news:.2%})")
        st.write(f"Bad News: {bad_count} ({bad_count/total_news:.2%})")
        st.write(f"Neutral News: {neutral_count} ({neutral_count/total_news:.2%})")
        if good_count > bad_count:
            st.success("The stock has a good ESG sentiment.")
        else:
            st.error("The stock has a bad ESG sentiment.")
    else:
        st.write("No news articles found for the given ticker.")

# Setup page configuration and main layout
st.set_page_config(page_title="PredictNTrade", layout="wide")

# Adding CSS for UI enhancements
st.markdown(
    """
    <style>
    body {
        background-color: #f7f9fc;
        color: #333;
        font-family: Arial, sans-serif;
    }
    .stApp {
        padding: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.25rem;
        color: #003366;
    }
    .css-18e3th9 {
        padding-top: 1rem;
    }
    .css-1d391kg {
        padding: 1rem;
    }
    .stHeader {
        font-size: 2rem;
        margin-bottom: 1rem;
        color: #003366;
    }
    .stSubheader {
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
        color: #003366;
    }
    .stButton button {
        background-color: #0047bb;
        color: white;
        padding: 10px 24px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-size: 1.25rem;
    }
    .stButton button:hover {
        background-color: #003366;
    }
    .stTextInput, .stSelectbox, .stDataFrame {
        background-color: #ffffff;
        color: #333;
        border: 1px solid #ccc;
        border-radius: 4px;
        padding: 8px;
    }
    .stDataFrame {
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .header-bar {
        background-color: #0047bb;
        padding: 1rem;
        color: white;
        font-size: 1.5rem;
        text-align: center;
        border-bottom: 4px solid #003366;
    }
    .section-title {
        font-size: 1.5rem;
        color: #003366;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .divider {
        border-top: 1px solid #e0e0e0;
        margin: 1.5rem 0;
    }
    .welcome-section {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin-bottom: 2rem;
    }
    .features-section {
        background-color: #f0f4f7;
        padding: 2rem;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-box {
        display: inline-block;
        width: 30%;
        margin: 1%;
        padding: 1rem;
        background-color: #ffffff;
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .feature-title {
        font-size: 1.25rem;
        color: #003366;
        margin-bottom: 0.5rem;
    }
    .feature-description {
        font-size: 1rem;
        color: #666;
    }
    .home-button-container {
        text-align: center;
        margin-top: 2rem;
    }
    .home-button {
        background-color: #0047bb;
        color: white;
        padding: 10px 24px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-size: 1.25rem;
        text-decoration: none;
    }
    .home-button:hover {
        background-color: #003366;
    }
    .sidebar .sidebar-content {
        background-color: #e3f2fd;
        padding: 1rem;
    }
    .sidebar .sidebar-title {
        font-size: 1.5rem;
        color: #0047bb;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sidebar .sidebar-button {
        background-color: #0047bb;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 10px;
        font-size: 1rem;
        margin-bottom: 0.5rem;
        width: 100%;
        text-align: left;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .sidebar .sidebar-button:hover {
        background-color: #003366;
    }
    .sidebar .sidebar-divider {
        border-top: 1px solid #cccccc;
        margin: 1rem 0;
    }
    .header {
        font-size: 2rem;
        color: #0047bb;
        font-weight: bold;
    }
    .input-label {
        font-size: 1rem;
        color: #333333;
        font-weight: bold;
    }
    .input-field {
        background-color: #ffffff;
        color: #333;
        border: 1px solid #ccc;
        border-radius: 4px;
        padding: 8px;
        width: 100%;
        margin-bottom: 1rem;
    }
    .data-table {
        font-size: 1rem;
        color: #333333;
    }
    .btn {
        background-color: #0047bb;
        color: white;
        font-size: 1rem;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .btn:hover {
        background-color: #003366;
    }
    .center {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Main rendering functions
def render_home():
    st.markdown('<div class="header-bar">PredictNTrade ESG Score Analysis and Forecasting Tool</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="welcome-section">
            <h1>Welcome to the PredictNTrade ESG Score Analysis and Forecasting Tool</h1>
            <p>This tool provides insights into the ESG performance of companies, along with financial analysis and forecasting.</p>
            <div class="center">
                <img src="data:image/png;base64,{}" alt="London Stock Exchange Group">
            </div>
        </div>
        """.format(base64.b64encode(open("/Users/ghadielhayek/Desktop/Ghadi's hackathon/LSE_Group_logo.svg.png", "rb").read()).decode()),
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="features-section">
            <div class="feature-box">
                <div class="feature-title">Comprehensive ESG Analysis</div>
                <div class="feature-description">Evaluate companies based on environmental, social, and governance criteria to understand their ESG performance.</div>
            </div>
            <div class="feature-box">
                <div class="feature-title">Financial Indicators</div>
                <div class="feature-description">Analyze financial data using various technical indicators such as Moving Averages, Bollinger Bands, MACD, RSI, and Fibonacci Retracement.</div>
            </div>
            <div class="feature-box">
                <div class="feature-title">Insights and Forecasting</div>
                <div class="feature-description">Generate insights and forecast future ESG scores using advanced machine learning models.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button('Go to Dashboard'):
        st.session_state['page'] = 'main'

def render_main():
    st.sidebar.markdown('<div class="sidebar"><h2 class="sidebar-title">Navigation</h2>', unsafe_allow_html=True)
    if st.sidebar.button("ESG Predictor", key='esg', help='Predict ESG scores for companies'):
        st.session_state['page'] = 'esg_predictor'
    if st.sidebar.button("Technical Analysis", key='tech', help='Perform technical analysis on stock data'):
        st.session_state['page'] = 'technical_analysis'
    if st.sidebar.button("Sentiment News Predictor", key='news', help='Analyze news sentiment for companies'):
        st.session_state['page'] = 'sentiment_news'
    if st.sidebar.button("Build Your Portfolio", key='portfolio', help='Create and manage your investment portfolio'):
        st.session_state['page'] = 'portfolio_builder'
    st.sidebar.markdown('<div class="sidebar-divider"></div><button class="sidebar-button" onclick="window.location.reload();">Home</button></div>', unsafe_allow_html=True)

    if st.session_state.get('page') == 'esg_predictor':
        render_esg_predictor()
    elif st.session_state.get('page') == 'technical_analysis':
        render_technical_analysis()
    elif st.session_state.get('page') == 'sentiment_news':
        render_sentiment_news()
    elif st.session_state.get('page') == 'portfolio_builder':
        render_portfolio_builder()

def render_esg_predictor():
    st.header("ESG Score Predictor")
    ticker_input = st.text_input("Enter Ticker Symbol", "AAPL", key="ticker")
    start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365*5), key="start_date")
    end_date = st.date_input("End Date", value=datetime.now(), key="end_date")
    if st.button("Fetch Data", key="fetch_data"):
        data = fetch_data(ticker_input, start_date, end_date)
        if data is not None:
            st.write("ESG and Financial Performance Data:", data)
            predictions = perform_regression(data)
            st.write("Predicted ESG Scores for the next periods:", predictions)

            # Create a gauge chart for the latest predicted ESG score
            if not predictions.empty:
                latest_esg_score = predictions['Predicted ESG Score'].iloc[-1]
                gauge_fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = latest_esg_score,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Latest Predicted ESG Score"},
                    gauge = {'axis': {'range': [None, 100]},
                             'bar': {'color': "darkblue"},
                             'steps' : [
                                 {'range': [0, 50], 'color': "red"},
                                 {'range': [50, 75], 'color': "yellow"},
                                 {'range': [75, 100], 'color': "green"}
                             ],
                             'threshold' : {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': 90}}))
                st.plotly_chart(gauge_fig, use_container_width=True)
        else:
            st.error("No data available or error in fetching data.")
    if st.button("Back"):
        st.session_state['page'] = 'main'

def render_technical_analysis():
    st.header("Technical Analysis Dashboard")
    ticker = st.text_input("Enter Stock Ticker", "AAPL")
    if st.button("Fetch and Display Technical Indicators"):
        fetch_and_plot_technical_indicators(ticker)
    if st.button("Back"):
        st.session_state['page'] = 'main'

def render_sentiment_news():
    st.header("Sentiment News Analysis")
    company_name = st.text_input("Enter Company Name for News", "Apple Inc.")
    if st.button("Analyze News"):
        news_df = fetch_esg_news_yahoo(company_name)
        if not news_df.empty:
            file_path = f"/Users/ghadielhayek/Desktop/Ghadi's hackathon/{company_name}_news.csv"
            save_news_to_file(news_df, file_path)
            analyze_and_display_news(news_df)
        else:
            st.write("No relevant news found or error in fetching.")
    if st.button("Back"):
        st.session_state['page'] = 'main'

def render_portfolio_builder():
    st.header("Build Your Sustainable Portfolio")
    tickers = st.text_area("Enter stock tickers separated by commas (e.g., AAPL, GOOGL, TSLA)", "AAPL, GOOGL, TSLA").split(',')

    portfolio_data = []

    for ticker in tickers:
        ticker = ticker.strip()
        if ticker:
            # Fetch market data
            market_data = fetch_market_data_yf(ticker)
            closing_price = market_data.get("Closing Price", "N/A")
            market_cap = market_data.get("Market Cap", "N/A")
            pe_ratio = market_data.get("P/E Ratio", "N/A")

            portfolio_data.append({
                "Ticker": ticker,
                "Closing Price": closing_price,
                "Market Cap": market_cap,
                "P/E Ratio": pe_ratio,
                "ESG Score": ""
            })

    if portfolio_data:
        portfolio_df = pd.DataFrame(portfolio_data)
        portfolio_df.set_index("Ticker", inplace=True)

        for ticker in tickers:
            ticker = ticker.strip()
            esg_score = st.text_input(f"ESG Score for {ticker}", key=f"esg_{ticker}")
            if esg_score:
                portfolio_df.at[ticker, "ESG Score"] = esg_score

        st.write(portfolio_df)

    if st.button("Back"):
        st.session_state['page'] = 'main'

def go_back():
    st.session_state['page'] = 'main'

# Navigation control based on session state
if _name_ == "_main_":
    if st.session_state.get('page', 'home') == 'home':
        render_home()
    elif st.session_state['page'] == 'main':
        render_main()
    elif st.session_state['page'] == 'esg_predictor':
        render_esg_predictor()
    elif st.session_state['page'] == 'technical_analysis':
        render_technical_analysis()
    elif st.session_state['page'] == 'sentiment_news':
        render_sentiment_news()
    elif st.session_state['page'] == 'portfolio_builder':
        render_portfolio_builder()