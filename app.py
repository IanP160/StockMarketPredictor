import logging
import pandas as pd
import numpy as np
import flask_cors
from pandas import Timestamp
from flask import Flask, request, jsonify, render_template
from StockPredictor import (
    load_model, prep_data, set_training_data, set_test_data, create_predictors,
    predict, predict_tomorrow, backtest, create_precisionScore,
    create_benchmarkScore, add_horizons, drop, fetch_data, fetch_news_sentiment, addSentiment
)

# Configure logging
logging.basicConfig(level=logging.DEBUG)  # Set the logging level to debug to see all messages

from flask_cors import CORS

app = Flask(__name__, template_folder='.')
CORS(app, resources={r"/*": {"origins": "*"}})

model = load_model()

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

def log(message):
    logging.info(message)
    log_messages.append(message)

@app.route('/predict', methods=['POST'])
def predict_stock():
    try:

        global log_messages
        log_messages = []  # Reset log messages for each request
        data = request.get_json()
        symbol = data.get('symbol', '^GSPC')  # S&P 500 as default symbol
        period = data.get('period', 'max')    # Default to 1 month of data if not specified

        # Fetch and process stock data
        historical_data = fetch_data(symbol, period)
        if historical_data.empty:
            logging.info("No historical data found.")
            return jsonify({'error': 'No historical data available for the provided symbol.'}), 404
        else:
            logging.info("Historical data fetched successfully.")

        prepared_data = prep_data(historical_data)
        if prepared_data.empty:
            logging.info("No prepared data found.")
            return jsonify({'error': 'No prepared data available for the provided symbol.'}), 404
        else:
            logging.info("Prepared data fetched successfully.")

        sentiment_score = fetch_news_sentiment(symbol)


        prepared_data, new_predictors = add_horizons(prepared_data, [2, 5, 60, 250, 1000])
        if prepared_data.empty:
            logging.info("No prepared data found after adding horizons.")
            return jsonify({'error': 'No prepared data available for the provided symbol.'}), 404
        else:
            logging.info("Prepared data fetched successfully after adding horizons.")

        prepared_data, newer_predictors = addSentiment(prepared_data, symbol, new_predictors)
        if prepared_data.empty:
            logging.info("No prepared data found after adding sentiment.")
        else:
            logging.info("Prepared data fetched successfully after adding sentiment.")


        prepared_data = drop(prepared_data)
        if prepared_data.empty:
            logging.info("No prepared data found after dropping.")
            return jsonify({'error': 'No prepared data available for the provided symbol.'}), 404
        else:
            logging.info("Prepared data fetched successfully after dropping.")

        # Segment data
        train_data = set_training_data(prepared_data)
        test_data = set_test_data(prepared_data)
        if train_data.empty:
            logging.warning("Train data is empty after segmentation.")
        else:
            logging.info("Train data segmented successfully.")
        if test_data.empty:
            logging.warning("Test data is empty after segmentation.")
        else:
            logging.info("Test data segmented successfully.")
        
        predictors = create_predictors(prepared_data, newer_predictors)

    
        if not predictors:  # This will be True if predictors is an empty list
            logging.warning("Predictors are empty after creation.")
        else:
            logging.info("Predictors created successfully")

        # Make predictions
        if 'Sentiment' not in test_data.columns:
            logging.error("Sentiment feature is missing in the test dataset.")
        else:
            logging.info("Sentiment feature found in the test dataset.")
        current_predictions = predict(train_data, test_data, predictors, model)
        if current_predictions.empty:
            logging.warning("No predictions made.")
        else:
            logging.info("Predictions made successfully.")
        

        # Evaluate predictions
        if test_data.empty:
            logging.warning("Insufficient data for precision score evaluation.")
            precision_score_val = "N/A"
            benchmark_score = "N/A"
        else:
            precision_score_val = create_precisionScore(test_data, model, predictors)
            benchmark_score = create_benchmarkScore(prepared_data, model, predictors)

        # Backtest if feasible
        backtest_results = backtest(prepared_data, model, predictors) if not prepared_data.empty else {}
        if backtest_results.empty:
            logging.warning("No backtest results available.")
        else:
            logging.info("Backtest results generated successfully.")

        recommendations = predict_tomorrow(symbol, prepared_data, model, predictors)
        if not recommendations:  # This will check if recommendations is an empty string
            logging.warning("No recommendations made.")
        else:
            logging.info("Recommendations made successfully.")
        

        
        ##benchmark_score = benchmark_score.to_dict()

    

        current_predictions = convert_timestamps(current_predictions)
        check_for_timestamps(current_predictions)  # Debugging to ensure conversion
        predictions_dict = current_predictions.to_dict()

        
        recommendations = convert_timestamps(recommendations)
        check_for_timestamps(recommendations)  # Debugging to ensure conversion

        
        backtest_results = convert_timestamps(backtest_results)
        check_for_timestamps(backtest_results)  # Debugging to ensure conversion
        backtest_results_dict = backtest_results.to_dict()

        # Respond with comprehensive data
        response = jsonify({
            'symbol': symbol,
            'predictions': predictions_dict, 
            'recommendations': recommendations,
            'precision_score': precision_score_val,
            'benchmark_score': benchmark_score,  # Wrapping scalar in a dictionary
            'sentiment_score': sentiment_score,
            'logs': log_messages  # Include log messages in the response


            ##'backtest_results': backtest_results_dict
        })
        print(response.get_data(as_text=True))
        return response

    except Exception as e:
        logging.error(f"Error in predict_stock: {str(e)}")
        return jsonify({'error': str(e)}), 500

def convert_timestamps(obj):
    if isinstance(obj, pd.DataFrame):
        # Convert datetime columns to string
        for col in obj.select_dtypes(include=['datetime64[ns]', 'datetime']).columns:
            obj[col] = obj[col].astype(str)
        # Convert datetime index if present
        if isinstance(obj.index, pd.DatetimeIndex):
            obj.index = obj.index.astype(str)
        return obj

    elif isinstance(obj, pd.Series):
        # Convert the Series values if they are of datetime type
        if pd.api.types.is_datetime64_any_dtype(obj):
            obj = obj.astype(str)
        # Convert the Series index if it is of datetime type
        if isinstance(obj.index, pd.DatetimeIndex):
            obj.index = obj.index.astype(str)
    return obj


def check_for_timestamps(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            check_for_timestamps(value)
    elif isinstance(obj, list):
        for item in obj:
            check_for_timestamps(item)
    elif isinstance(obj, pd.DataFrame):
        if not obj.select_dtypes(include=[np.datetime64, 'datetime']).empty:
            print("DataFrame contains datetime data.")
    elif isinstance(obj, pd.Series):
        if pd.api.types.is_datetime64_any_dtype(obj):
            print("Series contains datetime data.")
    elif isinstance(obj, (pd.Timestamp, np.datetime64)):
        print("Object is a datetime object.")
    else:
        print("No datetime object found.")


if __name__ == '__main__':
    app.run(debug=True, port=5000)
