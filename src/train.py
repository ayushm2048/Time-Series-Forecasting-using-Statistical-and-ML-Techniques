import config
from data_loader import load_data, fetch_weather_data, merge_weather
from features import create_features, get_feature_columns
from evaluate import evaluate_model
from models.svr import SVRPipeline
from models.random_forest import RandomForestPipeline

def main():
    # ==========================================
    # SETTINGS
    # ==========================================
    # Change this variable to choose which model to run:
    # Options: 'svr', 'random_forest'
    MODEL_TO_RUN = 'svr'
    
    # Set to False if you don't want to use weather data
    USE_WEATHER = True
    # ==========================================
    
    print("Loading data...")
    df = load_data(config.DATA_PATH)
    
    if USE_WEATHER:
        weather_df = fetch_weather_data()
        df = merge_weather(df, weather_df)
        
    print("Engineering features...")
    df = create_features(df, with_weather=USE_WEATHER)
    
    # Split data based on configuration dates
    train = df[df.index < config.SPLIT_DATE_TRAIN_END].copy()
    test = df[df.index >= config.SPLIT_DATE_VAL_END].copy()
    # Note: Validation set can be added here using SPLIT_DATE_VAL_END
    
    features = get_feature_columns(with_weather=USE_WEATHER)
    
    X_train = train[features]
    y_train = train['Total Load (MW)']
    
    X_test = test[features]
    y_test = test['Total Load (MW)']
    
    # Initialize the selected model
    if MODEL_TO_RUN == 'svr':
        pipeline = SVRPipeline(kernel='rbf', C=1, gamma=0.1, epsilon=0.01)
    elif MODEL_TO_RUN == 'random_forest':
        pipeline = RandomForestPipeline(n_estimators=100)
    else:
        raise ValueError(f"Unknown model: {MODEL_TO_RUN}")
        
    # Train and evaluate
    pipeline.train(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    evaluate_model(y_test, y_pred, model_name=MODEL_TO_RUN.upper())

if __name__ == "__main__":
    main()
