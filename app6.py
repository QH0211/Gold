import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os

# Nombre del archivo para almacenar datos históricos
DATA_FILE = "historical_gold.csv"

def load_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE, parse_dates=["Fecha"])
    else:
        return pd.DataFrame(columns=["Fecha", "Cierre"])

def save_data(df):
    df.to_csv(DATA_FILE, index=False)

def add_daily_data(fecha, cierre):
    # Convertir la fecha a pd.Timestamp para que sea comparable con las existentes
    fecha = pd.Timestamp(fecha)
    df = load_data()
    new_row = pd.DataFrame({"Fecha": [fecha], "Cierre": [cierre]})
    df = pd.concat([df, new_row], ignore_index=True)
    # Eliminar duplicados (por fecha) conservando el último registro
    df = df.drop_duplicates(subset=["Fecha"], keep="last")
    df.sort_values("Fecha", inplace=True)
    save_data(df)
    return df

def create_sequences_multi(data, window_size, forecast_horizon):
    """
    Crea secuencias multi‑paso:
    - Cada entrada (X) es una ventana de tamaño window_size.
    - Cada salida (Y) es un vector con los siguientes forecast_horizon valores.
    """
    X, Y = [], []
    for i in range(window_size, len(data) - forecast_horizon + 1):
        X.append(data[i-window_size:i])
        Y.append(data[i:i+forecast_horizon])
    return np.array(X), np.array(Y)

def train_and_predict_lstm_multi(df, forecast_horizon, window_size=60, epochs=20):
    """
    Entrena un modelo LSTM para predicción multi‑paso directa.
    Utiliza la columna "Cierre" para entrenar un modelo que predice el siguiente día
    (forecast_horizon=1) a partir de una ventana de tamaño window_size.
    Compara el primer valor predicho con el último valor real para indicar si mañana subirá o bajará.
    Devuelve un DataFrame con la fecha del siguiente día, el precio predicho y un mensaje.
    """
    precios = df["Cierre"].values.reshape(-1, 1)
    if len(precios) < window_size + forecast_horizon:
        st.error(f"Se requieren al menos {window_size + forecast_horizon} registros. Actualmente hay {len(precios)}.")
        return None, None
    scaler = MinMaxScaler(feature_range=(0, 1))
    precios_scaled = scaler.fit_transform(precios)
    
    X, Y = create_sequences_multi(precios_scaled, window_size, forecast_horizon)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    Y_train, Y_test = Y[:train_size], Y[train_size:]
    
    # Redimensionar para LSTM: [samples, window_size, 1]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(window_size, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(forecast_horizon))
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X_train, Y_train, epochs=epochs, batch_size=32, validation_data=(X_test, Y_test), verbose=0)
    
    # Predicción multi‑paso usando la última ventana
    last_window = precios_scaled[-window_size:]
    last_window = last_window.reshape(1, window_size, 1)
    pred_scaled = model.predict(last_window, verbose=0).flatten()
    predictions = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    
    next_day_prediction = predictions[0]
    last_actual_price = df["Cierre"].iloc[-1]
    movement = "Subirá" if next_day_prediction > last_actual_price else "Bajará"
    next_day_date = df["Fecha"].max() + pd.Timedelta(days=1)
    
    pred_df = pd.DataFrame({
        "Fecha": [next_day_date],
        "Prediccion": [next_day_prediction],
        "Movimiento": [movement]
    })
    return pred_df, model

def main_app():
    st.title("Actualización Diaria y Predicción del Precio del Oro con LSTM")
    st.write("""
    Esta aplicación permite que, día a día, ingreses el precio real del oro y se actualice el histórico.
    Con estos datos se reentrena un modelo LSTM que, usando una ventana de datos, predice el precio del próximo día
    y te indica si se espera que el precio suba o baje.
    """)
    
    st.header("Agregar Registro Diario")
    with st.form("Agregar Registro Diario"):
        fecha = st.date_input("Fecha", value=pd.Timestamp.today())
        cierre = st.number_input("Precio del Oro (Cierre)", min_value=0.0, value=1800.0)
        submit_data = st.form_submit_button("Agregar Registro")
    if submit_data:
        df = add_daily_data(fecha, cierre)
        st.success("Registro agregado correctamente.")
        st.write("Datos actualizados:")
        st.dataframe(df)
    
    # Cargar datos combinados
    df = load_data()
    if df.empty:
        st.warning("No hay datos disponibles. Por favor, agrega registros diarios o carga un CSV histórico.")
        return
    
    st.header("Filtrar Datos para Entrenamiento")
    period_option = st.selectbox("Selecciona el periodo de datos",
                                 ["Usar todos los datos", "Últimos 3 años", "Últimos 5 años"])
    if period_option == "Últimos 3 años":
        start_date = df["Fecha"].max() - pd.DateOffset(years=3)
        df_filtered = df[df["Fecha"] >= start_date]
    elif period_option == "Últimos 5 años":
        start_date = df["Fecha"].max() - pd.DateOffset(years=5)
        df_filtered = df[df["Fecha"] >= start_date]
    else:
        df_filtered = df.copy()
    st.write("Datos para entrenamiento:")
    st.dataframe(df_filtered.head())
    
    st.header("Gráfico del Histórico de Precios")
    fig_hist, ax_hist = plt.subplots(figsize=(12, 6))
    ax_hist.plot(df_filtered["Fecha"], df_filtered["Cierre"], label="Precio Real", color="gold")
    ax_hist.set_title("Histórico del Precio del Oro")
    ax_hist.set_xlabel("Fecha")
    ax_hist.set_ylabel("Precio del Oro (USD)")
    ax_hist.legend()
    st.pyplot(fig_hist)
    
    st.header("Entrenar Modelo y Generar Predicción")
    window_size = st.number_input("Longitud de la ventana (días)", min_value=10, max_value=365, value=60)
    forecast_horizon = 1  # Para predecir solo el siguiente día
    epochs = st.number_input("Número de épocas", min_value=5, max_value=100, value=20)
    
    if st.button("Reentrenar y Predecir"):
        with st.spinner("Entrenando modelo LSTM y generando predicción..."):
            pred_df, model = train_and_predict_lstm_multi(df_filtered, forecast_horizon, window_size, epochs)
        if pred_df is not None:
            st.success("Predicción generada con éxito.")
            st.write("**Predicción para el próximo día:**")
            st.dataframe(pred_df)
            st.write(f"Alerta: Se predice que mañana el precio {pred_df['Movimiento'].iloc[0]}.")
            
            # Graficar histórico y la predicción
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df_filtered["Fecha"], df_filtered["Cierre"], label="Precio Real (Histórico)", color="gold")
            ax.scatter(pred_df["Fecha"], pred_df["Prediccion"], label="Predicción Próximo Día", color="blue", s=100)
            ax.set_title("Histórico y Predicción del Precio del Oro")
            ax.set_xlabel("Fecha")
            ax.set_ylabel("Precio del Oro (USD)")
            ax.legend()
            st.pyplot(fig)

if __name__ == "__main__":
    main_app()
