import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(y_true, y_pred, model_name="Model"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n--- {model_name} PERFORMANCE ---")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")
    return mae, rmse, r2

def plot_predictions(y_true, y_pred, index, title="Load Forecast", zoom_hours=168):
    plt.figure(figsize=(16, 6))
    plt.plot(index[:zoom_hours], y_true.iloc[:zoom_hours] if hasattr(y_true, 'iloc') else y_true[:zoom_hours], color='black', alpha=0.7, label='Actual Load')
    plt.plot(index[:zoom_hours], y_pred[:zoom_hours], color='blue', linestyle='--', linewidth=2, label='Forecast')
    plt.title(f'{title} (Zoomed: {zoom_hours} Hours)')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.show()
