**Overview**

This project aims to explore the volatility of stablecoins, specifically USDC/USDT, in a dynamic market. With the rise of stablecoins and their importance in decentralized finance, understanding their price behavior is crucial for investors and market participants. The goal is to predict positive price growth (target variable, y) using advanced machine learning techniques, providing valuable insights that can help market participants make informed decisions in a volatile environment

The repository contains Python scripts for fetching, processing, and saving cryptocurrency and stock market data. It uses APIs (Binance for crypto, Yahoo Finance for stocks) to retrieve hourly data, enrich it with calculated metrics, and export the data as CSV files for analysis.

The project includes models such as Linear Regression (LR), Decision Trees (DT), Random Forest (RF), and XGBoost.

```bash
stable_coin/  
├── images/                                     # Contains the images that are generated through EDA     
│   ├── boxplot_usdc.png      
│   ├── price_change_over_time.png      
│   ├── price_change_correlation_with_volume.png      
│   ├── distribution_price_change.png      
├── README.md                      
├── notebook.ipynb/       
│   ├── get_coins                               # Fetches and processes cryptocurrency data     
│   ├── get_stocks                              # Fetches and processes stock data      
│   ├── feature engineering                     # Adds derived metrics for ML models      
│   ├── model evaluation and tuning             # Compares models and saves the best as a pickle file      
├── train.py                                    # Trains the best model            
├── predict.py                                  # Flask application for making predictions       
├── requirements.txt                            # List of required Python packages       
├── environment.yml                             # Conda environment file     
├── LICENSE      
├── Dockerfile                                  # For containerized deployment    
```

**Data Exploration:**

## Boxplot for Closing Prices for USDC

To better understand the data distribution and identify potential outliers, a boxplot of the closing prices for USDC was generated:

![Boxplot](images/boxplot_usdc.png)

Key Insights from Data Exploration Boxplot of USDC Close Prices:    
USDC prices mostly stay close to 1, indicating good stability as a stablecoin.
Outliers around 0.999 and 1.001 could be caused by short-term market fluctuations or external events.
The narrow Interquartile Range (IQR) shows minimal price fluctuations, suggesting consistent trading.
Outlier Clusters:

Outliers appear in clusters, possibly due to market distortions or technical issues.
Consecutive outliers point to specific periods of market stress or external events, such as political factors (presidential election).


## Price Change over Time for USDC

The following plot shows the price change over time for USDC in the filtered period from August to October 2024:

![Price Change Over Time for USDC](images/price_change_over_time.png)
Key observations:     
Price fluctuates slightly around the $1 mark, confirming USDC's stablecoin nature. Periods of slight instability (e.g., price > $1.001 or < $0.999) may indicate unusual market activity or temporary imbalances. Spikes at specific points (e.g., late August, mid-September) might be linked to external events or liquidity shifts.


## Price Changes and its correlation with Volume for USDC

This histogram shows the distribution of price changes for USDC, highlighting the frequency of different price change values over the dataset.

![Price Change and its Correlation with Volume for USDC](images/price_change_correlation_with_volume.png)
Key Insights from Data Exploration Boxplot of USDC Close Prices:     
The scatter plot shows the relationship between price change and trading volume for USDC/USDT. Most price changes are centered around 0, indicating stable behavior, even with varying volumes.  
There is a notable outlier with very high volume (~4e8) and a significant price change (~0.08). No clear linear correlation between volume and price change is observed, suggesting external factors might influence price deviations more than volume alone.

## Distribution of Price Change for USDC

![Distribution of Price Change](images/distribution_price_change.png)

Key observations:    
Most price changes are concentrated around 0, indicating that USDC's price remains stable, as expected for a stablecoin. The high frequency of 0 price changes confirms minimal fluctuations for USDC in most cases. Symmetric distribution: Price changes are evenly spread on both sides, showing that deviations (positive and negative) are rare and balanced. Outliers: Smaller bars farther from 0 represent rare, larger price changes, which are atypical.  
This validates USDC's stability for that time frame. The rare outliers might be caused by extreme market events or technical issues.




**Datasets**     
Stablecoin Data:
[Link to Cryptocurrencies Dataset](https://drive.google.com/file/d/18IzkQYiodTNiIxmnG7lGrrdb-akB0C-l/view?usp=sharing)

Stock Data:
[Link to Stocks Dataset](https://drive.google.com/file/d/1d4PRGApTcuQaCAj16dOc9k79P3M2PaYF/view?usp=sharing)

Merged Data with Features:
[Link to merged Dataset](https://drive.google.com/file/d/1aImaDFQWnDEN1wliP5KTh2MwfqFSktEi/view?usp=sharing)



**Features**

Crypto Data Fetcher: Retrieves OHLC data for selected cryptocurrencies and stablecoins using the Binance API, with additional derived metrics and timezone conversion.
Stock Data Fetcher: Fetches hourly stock data for predefined tickers using Yahoo Finance, enriching the data with calculated metrics.
USDC/USDT Analysis: Focuses on analyzing the stablecoin USDC/USDT, examining factors influencing its price growth, and using machine learning models to predict positive growth (y).
Machine Learning Models: Implements models like Linear Regression (LR), Decision Trees (DT), Random Forest (RF), and XGBoost to predict price trends and growth of USDC/USDT.
Flask API: A Flask-based API is included to interact with the data programmatically (optional, for deployment).
Docker Support: A Dockerfile is provided for easy deployment in containerized environments.

### Machine Learning Models
The project employs various machine learning models to predict the one-hour price growth of USDC/USDT. The pipeline includes:

Feature Engineering: Derived metrics such as moving averages (7d, 30d), volatility, technical indicators (e.g., RSI, MACD), and others.

Models:

Linear Regression (LR)
Decision Trees (DT)
Random Forest (RF)
XGBoost (Primary Model)
Evaluation:

Metrics: Accuracy, Precision, Recall, and AUC.
Dataset Split: 80% for training, 20% for testing.
Validation: 5-fold cross-validation for XGBoost.


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
- Jupyter Notebook (notebook.ipynb):
Fetch cryptocurrency and stock market data:
Fetches data for cryptocurrencies and stablecoins defined in the coins list.
Processes the data and adds derived metrics (e.g., price change).
Saves the final dataset as stable_coins.csv.
Fetches hourly stock data for predefined tickers, adds derived metrics, and formats timestamps.
Combines all stock data into a single DataFrame and saves it as a CSV.
Logs missing or delisted stocks/cryptos as warnings or errors.
Perform feature engineering and derive metrics.
Evaluate multiple machine learning models.
Save the best models as .pkl files.
Train the Model:
Use train.py to train the best-performing model (default: XGBoost) on the processed data.
Save the trained model as a .pkl file.
Deploy the Model with Flask:
Use predict.py to deploy the model and provide predictions via a Flask API.

### Flask API
The repository includes a Flask API (`predict.py`) to interact with the trained XGBoost model. The API allows users to predict whether the price of USDC/USDT will grow positively within the next hour.

#### Steps to Use
1. **Start the Flask Server**  
   Ensure the conda environment is active and run:
   ```bash
   python predict.py
The API will start locally at http://0.0.0.0:8000.

2. **Make Predictions**
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
To simplify deployment, a Dockerfile is provided. To build and run the Docker container:

## Build the Docker image:
docker build -t crypto-stock-analysis .
Run the container:
docker run -p 5000:5000 crypto-stock-analysis


License
This project is open-source and licensed under the MIT License.