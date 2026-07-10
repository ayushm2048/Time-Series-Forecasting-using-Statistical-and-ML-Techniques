import pandas as pd

def add_time_features(df):
    df['Hour'] = df.index.hour
    df['DayOfWeek'] = df.index.dayofweek
    df['Month'] = df.index.month
    df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    return df

def add_lag_features(df):
    df['Load_24h_Ago'] = df['Total Load (MW)'].shift(24)
    df['Load_48h_Ago'] = df['Total Load (MW)'].shift(48)
    df['Load_168h_Ago'] = df['Total Load (MW)'].shift(168)
    return df

def add_weather_features(df):
    if 'temperature_2m' in df.columns:
        df['temp_sq'] = df['temperature_2m']**2
    if 'precipitation' in df.columns:
        df['rain_flag'] = (df['precipitation'] > 0).astype(int)
    return df

def create_features(df, with_weather=True):
    df = df.copy()
    df = add_time_features(df)
    df = add_lag_features(df)
    if with_weather:
        df = add_weather_features(df)
    return df.dropna()

def get_feature_columns(with_weather=True):
    features = ['Hour', 'DayOfWeek', 'Month', 'IsWeekend', 'Load_24h_Ago', 'Load_48h_Ago', 'Load_168h_Ago']
    if with_weather:
        features.extend(['temp_sq', 'rain_flag', 'temperature_2m', 'relative_humidity_2m', 'cloud_cover', 'wind_speed_10m'])
    return features
