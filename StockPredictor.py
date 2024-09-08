#!/usr/bin/env python
# coding: utf-8


# In[93]:
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from datetime import datetime, timedelta
from textblob import TextBlob  # For sentiment analysis
import requests  # For fetching news data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def fetch_data(ticker_symbol="^GSPC", period="max"):
    """
    Fetch historical stock data using yfinance.
    :param ticker_symbol: Stock symbol, default is S&P 500 (^GSPC).
    :param period: Period for fetching data, default is 'max' for maximum historical data.
    :return: A DataFrame with the stock data.
    """

    try:
        stock = yf.Ticker(ticker_symbol)
        stock_data = stock.history(period=period)
        if stock_data.empty:
            logging.error(f"No data fetched for {ticker_symbol} with period {period}")
            raise ValueError("Data fetch resulted in an empty DataFrame")
        logging.info(f"Data fetched successfully for {ticker_symbol}")
        return stock_data
    except Exception as e:
        logging.error(f"Error fetching data for {ticker_symbol}: {str(e)}")
        raise



def prep_data(stock, start_date=None):
    try:
        if start_date is None:
            start_date = stock.index.min() if not stock.empty else None
        if 'Close' not in stock.columns:
            raise ValueError("Required column 'Close' is missing from the data")
        stock = stock.loc[start_date:].copy().dropna()
        
        stock["Tomorrow"] = stock["Close"].shift(-1)
        stock["Target"] = (stock["Tomorrow"] > stock["Close"]).astype(int)
        logging.info("Data preprocessed successfully")
        return stock
    except Exception as e:
        logging.error(f"Error in prep_data: {str(e)}")
        raise    





    
def set_training_data(stock):
    train = stock.iloc[:-100].copy()
    if train.empty:
        raise ValueError("Training data is empty")
    return train


def set_test_data(stock):
    test = stock.iloc[-100:].copy()
    if test.empty:
        raise ValueError("Test data is empty")
    return test

def create_predictors(stock, additional_predictors=[]):
    base_predictors = ["Close", "Volume", "Open", "High", "Low"]
    return base_predictors + additional_predictors


def load_model():
    model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)
    return model


# In[131]:


def predict(train, test, predictors, model):

    if train.empty or test.empty:
        logging.error("Training or test data is empty. Cannot proceed with model training.")
        return pd.DataFrame()  # Early exit with empty DataFrame if training or test data is empty.
    try:
        model.fit(train[predictors], train["Target"])
        preds = model.predict_proba(test[predictors])[:, 1]
        preds = (preds >= 0.6).astype(int)
        preds = pd.Series(preds, index=test.index, name="Predictions")
        
        # Check if predictions or test targets are empty
        if preds.empty or test["Target"].empty:
            logging.error("Predictions series or test target is empty, cannot concatenate.")
            return pd.DataFrame()

        combined = pd.concat([test["Target"], preds], axis=1)
  
        return combined
    except Exception as e:
        logging.error(f"Error during prediction or concatenation: {str(e)}")
        return pd.DataFrame()

def predict_and_recommend(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds_proba = model.predict_proba(test[predictors])[:, 1]
    recommendations = pd.Series(preds_proba, index=test.index, name="Recommendations")

    # Determine buy/sell based on a threshold
    recommendations = recommendations.apply(lambda x: 'Buy' if x >= 0.6 else 'Sell')
    return recommendations

def backtest(data, model, predictors, start=2500,step=250):
    all_predictions = []
    # Check if the dataset is smaller than the initial 'start' index
    if data.shape[0] < start:
        # Adjust 'start' to be smaller but still try to use a reasonable subset of data
        start = max(0, data.shape[0] - step)
        logging.info(f"Adjusted start index to: {start} due to small dataset.")

    # Check if 'start + step' is still greater than the total data length
    if start + step > data.shape[0]:
        # Reduce 'step' size if possible, or adjust 'start' again
        if start > step:
            start -= step  # Attempt to move start back to allow for one step
        else:
            step = max(50, data.shape[0] - start)  # Reduce step size to fit the data
        logging.info(f"Further adjusted start to: {start} and step to: {step}")
    count = step
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:count].copy()
      
        test = data.iloc[i:(i + step)].copy()
    
        if not train.empty and not test.empty:
            predictions = predict(train, test, predictors, model)
            
            if not predictions.empty:
                all_predictions.append(predictions)
                logging.debug(f"Added predictions for window from {i} to {i + step}.")
                
            else:
                logging.warning(f"No predictions generated for window from {i} to {i + step}.")
                break
        else:
            logging.info(f"Skipping window from {i} to {i + step} due to insufficient data. Details - Train size: {train.shape[0]}, Test size: {test.shape[0]}.")
            continue
        count += step


    if not all_predictions:
        logging.error("No valid predictions generated during backtesting.")
        return pd.DataFrame()  # Return an empty DataFrame if no predictions were made

    return pd.concat(all_predictions)


def create_precisionScore(stock, model, predictors):
    predictions = backtest(stock, model, predictors)
    if predictions.empty or ("Target" not in predictions.columns or "Predictions" not in predictions.columns):
        return "Insufficient data or missing columns for calculation"
    # Use zero_division=0 to handle cases where there are no predicted positive samples
    return precision_score(predictions["Target"], predictions["Predictions"], zero_division=0)

def create_benchmarkScore(stock, model, predictors):
    if stock.empty or "Target" not in stock.columns:
        return "No data to compute benchmark score"
    value_counts = stock["Target"].value_counts(normalize=True)
    benchmark_score = value_counts.get(1, 0.0)  # Return 0.0 if '1' is not present
    return benchmark_score


def add_horizons(stock, horizons = [2,5,60,250,1000]):
    new_predictors = []

    for horizon in horizons:
        rolling_averages = stock.rolling(horizon).mean()

        ratio_column = f"Close_Ratio_{horizon}"
        stock[ratio_column] = stock["Close"] / rolling_averages["Close"]

        trend_column = f"Trend_{horizon}"
        stock[trend_column] = stock.shift(1).rolling(horizon).sum()["Target"]
        new_predictors += [ratio_column, trend_column]

    logging.debug(f"Returning from add_horizons: {len(new_predictors)} predictors")
    return stock, new_predictors  # Return the updated DataFrame and the list of new predictors

def drop(stock):
    return stock.dropna()





## New Methods



def fetch_news_sentiment(ticker):
    # Your News API key
    api_key = 'e679b05ce59544a39886290a5b7a0fac'
    # Today's date
    today = datetime.now().strftime('%Y-%m-%d')
    # News API URL
    url = f"https://newsapi.org/v2/everything?q={ticker}&from={today}&sortBy=publishedAt&apiKey={api_key}"
    response = requests.get(url).json()

    # Calculate the average sentiment polarity of the news articles
    sentiment_score = 0
    articles = response.get('articles', [])
    if articles:
        total_polarity = sum(TextBlob(article['description']).sentiment.polarity for article in articles if article['description'])
        sentiment_score = total_polarity / len(articles) if articles else 0

    return sentiment_score

def addSentiment(prepared_Data, ticker, base_predictors):
    sentiment = fetch_news_sentiment(ticker)
    prepared_Data['Sentiment'] = sentiment
    predictors = base_predictors + ['Sentiment']
    return prepared_Data, predictors


def predict_tomorrow(ticker, prepared_Data, model, base_predictors):
    # Fetch the latest available data
    latest_data = prepared_Data
    # Calculate sentiment
    # Prepare the latest data for prediction
    # Create a new list of predictors including Sentiment
    # Extract the required features for the last available data point
    features = latest_data.tail(1)[base_predictors]
    # Predict the next day's movement
    prediction = model.predict(features)
    # Recommendation based on the prediction
    recommendation = 'Buy' if prediction[0] == 1 else 'Sell'
    return recommendation