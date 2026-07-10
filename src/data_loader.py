import pandas as pd
import requests

def load_data(filepath):
    df = pd.read_csv(filepath)
    if 'Date' in df.columns and 'Time' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')
    elif 'Datetime' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
        
    df = df.dropna(subset=['Datetime'])
    df = df.drop_duplicates(subset=['Datetime'], keep='first')
    df.set_index('Datetime', inplace=True)
    df.index = df.index.tz_localize(None)
    
    if df['Total Load (MW)'].dtype == 'object':
        df['Total Load (MW)'] = df['Total Load (MW)'].astype(str).str.replace(',', '')
    df['Total Load (MW)'] = pd.to_numeric(df['Total Load (MW)'], errors='coerce')
    
    df = df.asfreq('1H').ffill()
    return df

def fetch_weather_data(start_date="2023-12-01", end_date="2025-11-20"):
    print("Fetching weather data from Open-Meteo...")
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        "?latitude=12.75"
        "&longitude=78.34"
        f"&start_date={start_date}"
        f"&end_date={end_date}"
        "&hourly=temperature_2m,relative_humidity_2m,"
        "precipitation,cloud_cover,wind_speed_10m"
    )
    wdata = requests.get(url).json()
    weather = pd.DataFrame(wdata["hourly"])
    weather["Datetime"] = pd.to_datetime(weather["time"])
    weather["Datetime"] = weather["Datetime"].dt.tz_localize(None)
    weather.set_index("Datetime", inplace=True)
    weather = weather.sort_index()
    weather.drop(columns=['time'], inplace=True, errors='ignore')
    return weather

def merge_weather(df, weather_df):
    df = df.join(weather_df, how='left')
    df = df.ffill()
    return df
