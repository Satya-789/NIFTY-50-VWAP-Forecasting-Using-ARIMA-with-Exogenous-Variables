# NIFTY 50 VWAP Forecasting Using ARIMA with Exogenous Variables

## 📌 Project Overview

This project focuses on forecasting the **Volume Weighted Average Price (VWAP)** of the **NIFTY 50 index** using time-series forecasting techniques.

The project uses an **ARIMA model with exogenous variables (ARIMAX)**, implemented using the `pmdarima` library. Historical market data is transformed into additional rolling statistical features, which are then used as external predictors to improve VWAP forecasting performance.

The project demonstrates an end-to-end machine learning workflow, including:

* Data loading and preprocessing
* Missing-value handling
* Exploratory data analysis
* Rolling feature engineering
* Time-series train-test splitting
* Automated ARIMA model selection
* Forecast generation
* Model evaluation
* Visualization of actual vs. predicted VWAP
* Statistical model summary and performance analysis

---

## 🎯 Objective

The primary objective of this project is to predict future **NIFTY 50 VWAP values** using historical market information.

The target variable is:

* **VWAP** — Volume Weighted Average Price

The model uses historical market variables and their rolling statistics as exogenous predictors.

---

## 📂 Dataset

The project uses the `NIFTY50_all.csv` dataset.

The dataset contains historical NIFTY 50 market data, including variables such as:

* Date
* Open
* High
* Low
* Close
* VWAP
* Volume
* Turnover
* Trades

The `Date` column is used as the DataFrame index for time-series analysis.

> **Note:** The dataset file is not included in this repository. Place the CSV file in the appropriate local directory before running the notebook or Python script.

---

## 🛠️ Technologies Used

* **Python**
* **Pandas** — Data manipulation and preprocessing
* **NumPy** — Numerical computations
* **Matplotlib** — Data visualization
* **Scikit-learn** — Model evaluation metrics
* **pmdarima** — Automated ARIMA model selection
* **ARIMA / ARIMAX** — Time-series forecasting

---

## 🔄 Project Workflow

The overall workflow followed in this project is:

```text
Raw NIFTY 50 Dataset
        │
        ▼
Data Loading
        │
        ▼
Date as Index
        │
        ▼
Missing Value Analysis
        │
        ▼
Missing Value Removal
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Rolling Feature Engineering
        │
        ▼
Train-Test Split
        │
        ▼
Auto ARIMA Model Selection
        │
        ▼
ARIMAX Model Training
        │
        ▼
VWAP Forecasting
        │
        ▼
Model Evaluation
        │
        ▼
Actual vs Forecast Visualization
```

---

## 🧹 Data Preprocessing

The dataset is first loaded using Pandas:

```python
df = pd.read_csv("NIFTY50_all.csv")
```

The `Date` column is set as the index:

```python
df.set_index("Date", inplace=True)
```

Missing values are checked and removed:

```python
df.isna().sum()
df.dropna(inplace=True)
```

This ensures that the model receives clean and complete data.

---

## 📊 Exploratory Data Analysis

The project examines relationships between the major price variables:

```python
data[["Open", "High", "Low", "Close"]].corr()
```

The correlation matrix helps understand the relationship between the Open, High, Low, and Close prices.

The VWAP time series is also visualized:

```python
df["VWAP"].plot()
```

This provides an initial understanding of the behavior and trend of VWAP over time.

---

## ⚙️ Feature Engineering

To capture short-term market trends and volatility, rolling statistical features are created.

The following variables are used as lag-related features:

```python
lag_features = [
    "High",
    "Low",
    "Volume",
    "Turnover",
    "Trades"
]
```

Two rolling windows are considered:

* **3-day rolling window**
* **7-day rolling window**

### Rolling Mean

For each feature, 3-day and 7-day rolling averages are calculated.

Example:

```python
data[feature + "rolling_mean_3"] = (
    data[feature].rolling(window=3).mean()
)

data[feature + "rolling_mean_7"] = (
    data[feature].rolling(window=7).mean()
)
```

### Rolling Standard Deviation

Rolling standard deviation is also calculated to capture short-term volatility:

```python
data[feature + "rolling_std_3"] = (
    data[feature].rolling(window=3).std()
)

data[feature + "rolling_std_7"] = (
    data[feature].rolling(window=7).std()
)
```

These engineered variables provide the forecasting model with additional information about recent price behavior, trading activity, and market volatility.

---

## 🧮 Model Features

The model uses 20 engineered exogenous variables.

### Rolling Mean Features

* Highrolling_mean_3
* Highrolling_mean_7
* Lowrolling_mean_3
* Lowrolling_mean_7
* Volumerolling_mean_3
* Volumerolling_mean_7
* Turnoverrolling_mean_3
* Turnoverrolling_mean_7
* Tradesrolling_mean_3
* Tradesrolling_mean_7

### Rolling Standard Deviation Features

* Highrolling_std_3
* Highrolling_std_7
* Lowrolling_std_3
* Lowrolling_std_7
* Volumerolling_std_3
* Volumerolling_std_7
* Turnoverrolling_std_3
* Turnoverrolling_std_7
* Tradesrolling_std_3
* Tradesrolling_std_7

The target variable is:

```text
VWAP
```

---

## 🧪 Train-Test Split

The processed dataset is divided into training and testing datasets.

```python
training_data = data[0:1800]
test_data = data[1800:]
```

The first **1,800 observations** are used for training, while the remaining observations are reserved for testing.

This approach preserves the chronological order of the data, which is essential for time-series forecasting.

Unlike random train-test splitting, this method prevents future observations from being used to train the model.

---

## 🤖 ARIMAX Model

The project uses `auto_arima()` from the `pmdarima` library to automatically identify a suitable ARIMA configuration.

```python
from pmdarima import auto_arima

model = auto_arima(
    y=training_data["VWAP"],
    X=training_data[ind_features],
    trace=True
)
```

The model is then fitted using:

* **Target:** VWAP
* **Exogenous variables:** Engineered rolling features

```python
model.fit(
    y=training_data["VWAP"],
    X=training_data[ind_features]
)
```

This approach combines:

* **ARIMA** — captures temporal dependencies in VWAP
* **Exogenous variables** — incorporates additional market information

---

## 🔮 Forecasting

After training, VWAP values are forecasted for the test period:

```python
forecast = model.predict(
    n_periods=len(test_data),
    X=test_data[ind_features]
)
```

The forecasted values are then added to the test dataset:

```python
test_data["Forecast_ARIMA"] = forecast.values
```

---

## 📈 Model Evaluation

The model is evaluated using multiple regression and forecasting metrics.

### 1. Root Mean Squared Error (RMSE)

```python
np.sqrt(
    mean_squared_error(
        test_data["VWAP"],
        test_data["Forecast_ARIMA"]
    )
)
```

RMSE measures the average magnitude of prediction errors while giving greater weight to larger errors.

---

### 2. Mean Absolute Error (MAE)

```python
mean_absolute_error(
    test_data["VWAP"],
    test_data["Forecast_ARIMA"]
)
```

MAE represents the average absolute difference between actual and predicted VWAP values.

---

### 3. Mean Absolute Percentage Error (MAPE)

```python
mape = mean_absolute_percentage_error(
    test_data["VWAP"],
    test_data["Forecast_ARIMA"]
)

print(mape * 100)
```

MAPE expresses the forecasting error as a percentage, making it easier to interpret the relative prediction accuracy.

---

### 4. R² Score

```python
r2 = r2_score(
    test_data["VWAP"],
    test_data["Forecast_ARIMA"]
)

print(r2)
```

The R² score measures how much of the variation in the test-set VWAP is explained by the predictions.

---

## 📉 Actual vs Forecast Visualization

The actual and predicted VWAP values are plotted to visually evaluate model performance.

```python
plt.figure(figsize=(12, 5))

plt.plot(
    test_data["VWAP"],
    label="Actual"
)

plt.plot(
    test_data["Forecast_ARIMA"],
    label="Forecast"
)

plt.legend()
plt.show()
```

The visualization helps identify:

* How closely forecasts follow actual VWAP movements
* Periods of high forecasting error
* Whether the model captures overall trends
* Whether the model reacts adequately to changes in market behavior

---

## 📊 Model Summary

The ARIMA model summary is generated using:

```python
print(model.summary())
```

This provides information about the selected ARIMA configuration and estimated model parameters.

---

## 📁 Suggested Project Structure

```text
NIFTY50-VWAP-Forecasting/
│
├── Dataset/
│   └── NIFTY50_all.csv
│
├── Notebook/
│   └── NIFTY50_VWAP_Forecasting.ipynb
│
├── README.md
│
└── requirements.txt
```

---

## 📦 Installation

Install the required Python libraries using:

```bash
pip install pandas numpy matplotlib scikit-learn pmdarima
```

Or create a `requirements.txt` file:

```text
pandas
numpy
matplotlib
scikit-learn
pmdarima
```

Then install dependencies with:

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

1. Clone or download the project.
2. Place `NIFTY50_all.csv` inside the `Dataset` directory.
3. Update the dataset path in the Python code if required.
4. Install the required dependencies.
5. Run the Jupyter Notebook or Python script.
6. Review the generated:

   * Forecast values
   * RMSE
   * MAE
   * MAPE
   * R² score
   * Actual vs Forecast plot
   * ARIMA model summary

---

## ⚠️ Important Considerations

This project is intended for **educational and portfolio purposes**.

Financial markets are highly dynamic and influenced by numerous external factors. Historical price patterns and technical indicators do not guarantee future performance.

The model's predictions should **not be considered financial advice or a guaranteed trading strategy**.

Potential improvements could include:

* Hyperparameter optimization
* Walk-forward validation
* Time-series cross-validation
* Stationarity testing
* Residual diagnostics
* Feature selection
* Incorporating macroeconomic variables
* Adding technical indicators
* Comparing ARIMAX with Prophet, LSTM, XGBoost, or other forecasting models
* Developing a baseline model for comparison

---

## 🚀 Future Improvements

Future versions of this project could explore:

1. **Walk-Forward Forecasting**

   * Retrain or update the model periodically as new market data becomes available.

2. **Model Comparison**

   * Compare ARIMAX against:

     * SARIMA
     * Prophet
     * XGBoost
     * Random Forest
     * LSTM
     * GRU

3. **Advanced Feature Engineering**

   * RSI
   * MACD
   * Bollinger Bands
   * Moving averages
   * Momentum indicators

4. **Enhanced Validation**

   * Use rolling-window or expanding-window validation instead of a single train-test split.

5. **Residual Analysis**

   * Analyze model residuals to determine whether meaningful patterns remain unexplained.

6. **Trading Strategy Evaluation**

   * Evaluate whether forecasts can be converted into a systematic trading strategy using appropriate backtesting methodology.

---

## 👨‍💻 Author

**Satya Prakash Mohanty**

This project was developed as part of a portfolio focused on **Data Science, Machine Learning, and Time-Series Forecasting**.

---

## ⭐ Key Takeaway

This project demonstrates how **ARIMA-based time-series forecasting can be enhanced with exogenous market features**. By combining historical VWAP behavior with rolling statistics derived from price and trading activity, the project builds a structured forecasting pipeline for NIFTY 50 market data.

The project provides practical experience in **time-series preprocessing, feature engineering, ARIMAX modeling, forecasting, and quantitative model evaluation**.

