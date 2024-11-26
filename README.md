**Overview**

This project aims to provide insights into the factors affecting stablecoin USDC/USDT price movements and explore which features have an impact on its price growth (positive growth as the target variable, y). It aims to predict positive price growth using advanced ML techniques. This analysis can help market participants make informed decisions in a volatile environment.

This repository contains Python scripts for fetching, processing, and saving cryptocurrency and stock market data. The scripts utilize APIs (Binance for crypto, Yahoo Finance for stocks) to retrieve hourly data, enrich it with calculated metrics, and export the data as CSV files for analysis.

The project includes models such as Linear Regression (LR), Decision Trees (DT), Random Forest (RF), and XGBoost.

project/
├── notebook.ipynb/ 
│   ├── get_coins
│   ├── get_stocks
│   ├── feature engeneering
│   ├── model evaluation      # best models saved as pickle file
├── train.py                  # best model is trained 
├── pedict.py                 # flask application example
├── requirements.txt          # List of required Python packages
├── environment.yml           # Conda environment file
├── LICENSE  
├── Dockerfile                 


**Features**

Crypto Data Fetcher: Retrieves OHLC data for selected cryptocurrencies and stablecoins using the Binance API, with additional derived metrics and timezone conversion.
Stock Data Fetcher: Fetches hourly stock data for predefined tickers using Yahoo Finance, enriching the data with calculated metrics.
USDC/USDT Analysis: Focuses on analyzing the stablecoin USDC/USDT, examining the factors influencing its price growth and using machine learning models to predict positive growth (y).
Machine Learning Models: Implements models like Linear Regression (LR), Decision Trees (DT), Random Forest (RF), and XGBoost to predict price trends and growth of USDC/USDT.
Flask API: A Flask-based API is included to interact with the data programmatically (optional, for deployment).
Docker Support: Dockerfile is provided for easy deployment in containerized environments.

### Machine Learning Models
The project employs various machine learning models to predict the one-hour price growth of USDC/USDT. The pipeline includes:
- **Feature Engineering**: Derived metrics such as moving averages (7d, 30d), volatility, technical indicators (e.g., RSI, MACD), and others are included.
- **Models**:
  - Linear Regression (LR)
  - Decision Trees (DT)
  - Random Forest (RF)
  - XGBoost (primary model)
- **Evaluation**: 
  - Metrics: Accuracy, Precision, Recall, and AUC.
  - Dataset: Split into training (80%) and testing (20%) sets.
  - Validation: 5-fold cross-validation for XGBoost.


### Installation
1. Clone the repository
bash
git clone https://github.com/your-repo.git
cd your-repo
2. Set up the environment
Using Conda:

bash
conda env create -f environment.yml
conda activate your-environment-name
Using pip:

bash
pip install -r requirements.txt

### How to Use
- Use the Jupyter notebook (`notebook.ipynb`) to:
  - Fetch cryptocurrency and stock market data:
    The function:
    Logs the data retrieval process.
    Fetches data for cryptocurrencies and stablecoin defined in the coins list.
    Processes the data and adds derived metrics (e.g., price change).
    Saves the final dataset as stable_coins.csv.
    Fetches hourly stock data for a predefined list of tickers.
    Adds derived metrics and formats timestamps.
    Combines all stock data into a single DataFrame and saves it as a CSV.
    Missing or delisted stocks/cryptos are logged as warnings or errors.
  - Perform feature engineering and derive metrics.
  - Evaluate multiple machine learning models.
  - Save the best models as `.pkl` files.
- Use train.py to: Train a Machine Learning Model
    Train the best-performing model (default: XGBoost) on the processed data.
    Save the trained model as a .pkl file.
- Use predict.py to deploy the model and provide predictions via a Flask API.

### Flask API
The repository includes a Flask API (`predict.py`) to interact with the trained XGBoost model. The API allows users to predict whether the price of USDC/USDT will grow positively within the next hour.

#### Steps to Use
1. **Start the Flask Server**  
   Ensure the conda environment is active and run:
   ```bash
   python predict.py
The API will start locally at http://0.0.0.0:8000.

Make Predictions
Send an HTTP POST request with the input features as JSON to the /predict endpoint.
EInput Example:
{
    "30d_ma": 102.5,
    "7d_ma": 101.0,
    "ticker=BTCUSDT": 1,
    "ticker=AAPL": 0
}

Output Example:
{
    "is_positive_growth_1h_future_probability": 0.67,
    "is_positive_growth_1h_future": true
}


### Run with Docker
A Dockerfile is provided to simplify deployment. To build and run the Docker container:

## Build the Docker image:
docker build -t crypto-stock-analysis .
Run the container:
docker run -p 5000:5000 crypto-stock-analysis


Future Improvements
Add support for additional APIs and asset types.
Enable real-time streaming data.
Automate updates to the asset lists.

License
This project is open-source and licensed under the MIT License.