PredictNTrade

Overview

PredictNTrade is a comprehensive financial analytics platform designed to enhance investment strategies through advanced data analysis and prediction models. Developed with Streamlit, it integrates various data sources and analytical techniques to provide real-time insights into stock performance and market trends.

Features

ESG Score Predictor
Utilizes historical financial and ESG data from the Eikon API to predict future ESG scores of stocks using a RandomForestRegressor. This feature helps investors identify sustainable investment opportunities by forecasting the environmental, social, and governance performance of companies.

Technical Analysis Dashboard
Offers real-time technical analysis of stocks fetched from Yahoo Finance. It includes indicators such as Bollinger Bands, MACD, RSI, and SMA, visualized through interactive Plotly charts. This dashboard aids traders in making informed decisions based on current market conditions.

Sentiment News Analysis
Analyzes news sentiment related to specific companies using VADER from the NLTK library. It scrapes news articles from Yahoo Finance, classifies them into positive, negative, or neutral categories, and presents a sentiment overview. This analysis provides insights into how current news might affect stock performance.

Portfolio Builder
Allows users to construct and analyze a sustainable investment portfolio. It fetches market data for selected stocks via Yahoo Finance and lets users input ESG scores, facilitating informed and balanced investment choices based on both financial metrics and sustainability factors.

Technologies Used

Python: Core programming language
Streamlit: Framework for building the app interface
Plotly: For creating interactive visualizations
Yahoo Finance API: Real-time stock data
Eikon API: Historical financial and ESG data
NLTK: Natural language processing for sentiment analysis
Pandas & NumPy: Data manipulation and numerical analysis

