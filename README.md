# Power Grid Load Forecasting

This repository contains the end-to-end data pipeline and machine learning models for **Day-Ahead electricity load forecasting**. The project uses a continuous 2-year hourly dataset (Dec 2023 - Nov 2025) and incorporates external weather features to accurately predict power grid demand.

---

## Repository Structure

The repository has been neatly organized for readability and ease of use:

```text
├── data/
│   └── Hourly_Data_220kV.xlsx        
├── notebooks/
│   ├── 01_Random_Forest_Baseline.ipynb  
│   ├── 02_SVR_Model.ipynb             
│   └── 03_Stacked_LSTM_Model.ipynb     
├── README.md                           
```

---

## System Architecture & Pipeline

The forecasting pipeline involves the following core phases:

1. **Data Ingestion & Debugging** 🧹
   - Ingested ~17,300 hours of load data.
   - Handled duplicate timestamps and used forward-fill (`ffill`) for missing sensor gaps.

2. **External API Integration (Weather)** 🌦️
   - Integrated the **Open-Meteo Archive API**.
   - Joined chronological weather features (temperature, humidity, rain flag, cloud cover) to capture environmentally driven demand spikes (e.g., HVAC usage).

3. **Feature Engineering** ⚙️
   - **Time Normalization**: Extracted and scaled `Hour`, `DayOfWeek`, `Month`, and generated `IsWeekend`/`Holiday` binary flags.
   - **Lags**: Engineered strict Day-Ahead lag constraints (`24h`, `48h`, `168h` ago).

4. **Modeling & Inference** 🧠
   - Scaled inputs using `MinMaxScaler`/`StandardScaler`.
   - Iterated from statistical baselines (SARIMA) to machine learning (SVR) and deep learning (Stacked LSTM).
   - Designed a recursive look-ahead loop for 168-step (1-week) continuous forecasting.

---

## Key Results

Our pivot to **Support Vector Regression (SVR)** with an RBF kernel resulted in highly operational metrics on a strict, chronological 20% holdout test set:

- **R² Score:** `~0.84`
- **Mean Absolute Error (MAE):** `~9.26 MW`

---

## Setup & Installation

To run the notebooks locally:

1. Clone this repository.
2. Ensure you have `pandas`, `scikit-learn`, `tensorflow`/`keras`, and `jupyter` installed.
3. The notebooks are configured to read the dataset from the `../data/` directory relative to their new location inside `notebooks/`.
