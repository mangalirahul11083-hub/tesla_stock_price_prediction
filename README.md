# 🚗 Tesla Stock Price Prediction using SimpleRNN

This project is a machine learning web application built using TensorFlow and Streamlit to predict Tesla stock closing prices based on historical data using a SimpleRNN (Recurrent Neural Network) model.

The application allows users to:

- View historical Tesla stock prices 📊

- Predict future stock prices for the next N days 🔮

- Visualize prediction trends 📈

## 📌 Project Features

✅ Time-series forecasting using SimpleRNN

✅ Data preprocessing and MinMax scaling

✅ Sequence-based training (60-day window)

✅ Interactive web UI using Streamlit

✅ Graphs for historical and predicted prices

✅ Deployed on Streamlit Cloud

## 🧠 Model Architecture

SimpleRNN (50 units)

Dropout (0.2)

Dense (1 output neuron)

Optimizer: Adam

Loss: Mean Squared Error (MSE)

## 📂 Project Structure

```bash
tesla_stock_price_prediction/
│
├── app.py                     # Streamlit app
├── tesla_rnn_model.keras      # Trained RNN model
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
│
└── data/
    └── TSLA.csv               # Tesla stock dataset

```

## 🛠️ Installation & Setup (Local)
1️⃣ Clone the Repository

2️⃣ Create Virtual Environment (Optional)

3️⃣ Install Requirements

4️⃣ Run the App

## 📊 Dataset

Source: Tesla historical stock prices (Yahoo Finance or similar)

File: data/TSLA.csv

Column used: Close

## ⚠️ Disclaimer

- This project is for educational purposes only.

- It should not be used for real financial or trading decisions.

## ⭐ Acknowledgements

- TensorFlow / Keras

- Streamlit

- Scikit-learn

- Pandas & NumPy

## 🏁 Future Improvements

- Add LSTM / GRU models

- Add multi-feature input (Open, High, Low, Volume)

- Add model comparison dashboard

- Add confidence intervals

## ✅ Conclusion:

In this project, we successfully designed and deployed a Tesla Stock Price Prediction system using a SimpleRNN deep learning model. The model was trained on historical Tesla stock closing prices and uses time-series sequence learning to predict future price trends.

We implemented a complete end-to-end machine learning pipeline, including data preprocessing, normalization, sequence generation, model training, evaluation, and deployment using Streamlit. The interactive web application allows users to visualize historical stock prices and generate future predictions in an easy and user-friendly manner.

Although stock market prediction is inherently uncertain, this project demonstrates how Recurrent Neural Networks can effectively capture temporal patterns in financial time-series data. This system can be further enhanced by incorporating advanced models such as LSTM or GRU, additional market indicators, and more sophisticated evaluation techniques.

Overall, this project serves as a strong practical example of applying deep learning to real-world financial data and deploying machine learning models as web applications.
