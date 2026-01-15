import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="Tesla Stock Prediction", layout="centered")

st.title("🚗 Tesla Stock Price Prediction using SimpleRNN")
st.write("This app uses a trained SimpleRNN model to predict Tesla stock prices.")

# ----------------------------
# LOAD DATA
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/TSLA.csv")
    df = df[['Close']]
    return df

df = load_data()

# ----------------------------
# 📊 DATA VISUALIZATION SECTION
# ----------------------------
st.header("📊 Tesla Stock Price Analysis")

st.subheader("📈 Full Historical Closing Price")
st.line_chart(df["Close"])

st.subheader("📉 Last 100 Days Trend")
st.line_chart(df["Close"].tail(100))

# ----------------------------
# SCALE DATA
# ----------------------------
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df.values.reshape(-1, 1))

# ----------------------------
# CREATE SEQUENCES
# ----------------------------
def create_sequences(data, seq_length=60):
    X = []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
    return np.array(X)

SEQ_LEN = 60
X = create_sequences(scaled_data, SEQ_LEN)
X = X.reshape((X.shape[0], X.shape[1], 1))

last_sequence = X[-1].copy()

# ----------------------------
# BUILD MODEL (SAME AS NOTEBOOK)
# ----------------------------
def build_model():
    model = Sequential()
    model.add(SimpleRNN(50, input_shape=(60, 1)))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model

# ----------------------------
# LOAD WEIGHTS
# ----------------------------
@st.cache_resource
def load_trained_model():
    model = build_model()
    model.load_weights("tesla_rnn_model.keras")
    return model

model = load_trained_model()

# ----------------------------
# FUTURE PREDICTION FUNCTION
# ----------------------------
def predict_future(model, last_seq, days):
    future = []
    current_seq = last_seq.copy()

    for _ in range(days):
        pred = model.predict(current_seq.reshape(1, 60, 1), verbose=0)
        future.append(pred[0, 0])
        current_seq = np.append(current_seq[1:], pred)

    future = np.array(future).reshape(-1, 1)
    future = scaler.inverse_transform(future)
    return future

# ----------------------------
# 🔮 PREDICTION SECTION
# ----------------------------
st.header("🔮 Future Stock Price Prediction")

days = st.slider("Select number of days to predict:", 1, 30, 5)

if st.button("Predict"):
    predictions = predict_future(model, last_sequence, days)

    st.subheader("📌 Predicted Prices:")
    for i, price in enumerate(predictions, 1):
        st.write(f"Day {i}: ${price[0]:.2f}")

    # ----------------------------
    # 📉 COMPARISON CHART
    # ----------------------------
    st.subheader("📊 Prediction vs Recent History")

    last_60_days = df["Close"].tail(60).values
    combined = np.concatenate([last_60_days, predictions.flatten()])

    combined_df = pd.DataFrame({
        "Price": combined
    })

    st.line_chart(combined_df)

