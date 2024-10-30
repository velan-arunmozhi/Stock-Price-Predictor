# Stock Price Predictor
This project uses an LSTM (Long Short-Term Memory) model to predict the next day's closing price of Apple stock (AAPL) using historical stock market data. The model is designed to process sequences of stock prices to forecast future values and attempts to learn patterns in stock price movements.

## Project Overview
The notebook walks through the following steps:

### Data Collection:
The Yahoo! Finance API is used to get Apple's stock market data from the last 20 years. Only the closing prices are considered for training and evaluation of this model.

### Data Preprocessing:
1. Train/Test Split: The dataset is divided into 90% for training and 10% for testing.
2. Normalization: The closing prices are scaled to values between 0 and 1 using MinMaxScaler.
3. Sequence Generation: Sliding windows of 59 days are used to predict the 60th day's price. This is useful for the model to learn time-dependent patterns in the data.

### LSTM Model Architecture:
* Input Layer (Length of sequences (60) x 1)
* LSTM layer with 300 neurons
* Output layer for predicting the closing price

### Training and Evaluation:
The model is trained on the training set and tested on the holdout test set. Root mean squared error (RMSE) and mean absolute error (MAE) are used to evaluate the model's performance. Visualizations of the predicted vs. actual stock prices are shown to illustrate the model’s accuracy.

## Prerequisites
To run this notebook, you will need:
* Python 3.x

And the following libraries:
* yfinance
* pandas
* tensorflow
* keras
* scikit-learn
* matplotlib
* numpy

```
pip install yfinance pandas tensorflow keras scikit-learn matplotlib numpy
```

## Instructions
Clone or download this repository to your local machine.
Open the notebook (stock_pred_LSTM.ipynb) in Jupyter Notebook or Jupyter Lab.
You can modify parameters such as sequence length, model architecture, and epochs to experiment with different configurations

## Results
The trained model achieved a RMSE of ~3.0 and a MAE of ~2.3. Further improvments can be made by changing the number of epochs, using different model architectures, or training the model with more parameters.

## Acknowledgements
The stock data was obtained by using the Yahoo! Finance API. This project was inspired by this article (https://neptune.ai/blog/predicting-stock-prices-using-machine-learning) and a desire to apply deep learning to predict stock prices.
