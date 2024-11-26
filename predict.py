import pickle
import numpy as np
from flask import Flask, jsonify
import xgboost as xgb

# load model
with open('final_xgboost_model.pkl', 'rb') as file:
    model_xgb_final = pickle.load(file)

# Feature names
feature_names = [
    "30d_ma", "7d_ma", "7d_volatility", "adx", "aroon", "atr", "bop", "cci",
    "close", "cmo", "day", "fibonacci_0", "fibonacci_100", "fibonacci_23_6",
    "fibonacci_38_2", "fibonacci_50", "fibonacci_61_8", "growth_168h", "growth_24h",
    "growth_336h", "growth_48h", "growth_4h", "growth_720h", "growth_72h", "high",
    "hour", "intraday_range", "kama", "low", "macd", "macd_hist", "macd_signal", "mom",
    "month", "open", "ppo", "price_change", "roc", "rsi", "ticker=AAL", "ticker=AAPL",
    "ticker=ABEV", "ticker=ABT", "ticker=ADBE", "ticker=AGNC", "ticker=ALV", "ticker=AMD",
    "ticker=AMGN", "ticker=AMZN", "ticker=AVGO", "ticker=BA", "ticker=BAC", "ticker=BAK",
    "ticker=BBD", "ticker=BHP", "ticker=BKKT", "ticker=BLK", "ticker=BMY", "ticker=BNBUSDT",
    "ticker=BP", "ticker=BTCUSDT", "ticker=CAN", "ticker=CAT", "ticker=CCL", "ticker=CMCSA",
    "ticker=CME", "ticker=CMG", "ticker=COIN", "ticker=COST", "ticker=CRM", "ticker=CSCO",
    "ticker=CVX", "ticker=DHR", "ticker=DIS", "ticker=DJT", "ticker=DOCN", "ticker=DOGEUSDT",
    "ticker=DTE", "ticker=EBON", "ticker=ERIC", "ticker=ETHUSDT", "ticker=F", "ticker=GLD",
    "ticker=GOLD", "ticker=GOOG", "ticker=GOOGL", "ticker=GRAB", "ticker=GS", "ticker=HD",
    "ticker=HEI", "ticker=HON", "ticker=HUT", "ticker=IBM", "ticker=ICE", "ticker=INTC",
    "ticker=IONQ", "ticker=JBLU", "ticker=JNJ", "ticker=JOBY", "ticker=JPM", "ticker=KDP",
    "ticker=KGC", "ticker=LCID", "ticker=LIN", "ticker=LLY", "ticker=LMT", "ticker=MA",
    "ticker=MARA", "ticker=MDT", "ticker=MRK", "ticker=MS", "ticker=MSFT", "ticker=MSTR",
    "ticker=NDAQ", "ticker=NEM", "ticker=NFLX", "ticker=NIO", "ticker=NKE", "ticker=NOW",
    "ticker=NU", "ticker=NVDA", "ticker=NYCB", "ticker=OKLO", "ticker=PAYX", "ticker=PEP",
    "ticker=PFE", "ticker=PG", "ticker=PLTR", "ticker=PYPL", "ticker=QCOM", "ticker=RIOT",
    "ticker=RIVN", "ticker=SAP", "ticker=SBUX", "ticker=SMCI", "ticker=SNAP", "ticker=SOFI",
    "ticker=SOLUSDT", "ticker=SOUN", "ticker=SPGI", "ticker=SQ", "ticker=T", "ticker=TMO",
    "ticker=TSLA", "ticker=TWLO", "ticker=TXN", "ticker=UNH", "ticker=USDCUSDT", "ticker=V",
    "ticker=VALE", "ticker=VFC", "ticker=VZ", "ticker=WMT", "ticker=WULF", "ticker=XOM",
    "ticker_type=cryptocurrency", "ticker_type=stablecoin", "ticker_type=stock", "trix",
    "volume", "willr", "year"
]

# initialize flask
app = Flask(__name__)

# Define the prediction function
def predict_single(features, model):
    # reshaping for a single prediction
    X = np.array(features).reshape(1, -1) 
    # get probability for positive class 
    y_pred = model.predict_proba(X)[:, 1]  
    return y_pred[0]

# route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # simulate input
        partial_input = {
            "30d_ma": 102.5,
            "7d_ma": 101.0,
            "ticker=BTCUSDT": 1,
            "ticker=AAPL": 0,
        }

        # initialzie features
        features = [0] * len(feature_names)

        # set features
        for key, value in partial_input.items():
            if key in feature_names:
                features[feature_names.index(key)] = value

        # pred using model
        prediction = predict_single(features, model_xgb_final)
        
        # determine the prediction result, tershold 0.5
        positive_growth = prediction >= 0.5  
        
        # return json format
        result = {
            'is_positive_growth_1h_future_probability': float(prediction),
            'is_positive_growth_1h_future': bool(positive_growth)
        }
        return jsonify(result)  

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# run app
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)
