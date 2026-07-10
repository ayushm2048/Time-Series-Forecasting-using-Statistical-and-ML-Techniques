# Power Grid Load Forecasting

This repository contains the end-to-end data pipeline and machine learning models for Day-Ahead electricity load forecasting. The project uses a continuous 2-year hourly dataset and incorporates external weather features to predict power grid demand.

## Repository Structure

The codebase has been refactored into a modular Python architecture for production readiness and easy maintenance:

```text
├── data/
│   └── Hourly_Data_220kV.csv                  
├── notebooks/                                 
├── src/
│   ├── config.py                               
│   ├── data_loader.py                          
│   ├── evaluate.py                             
│   ├── features.py                             
│   ├── models/                                 
│   │   ├── random_forest.py
│   │   └── svr.py
│   └── train.py                            
```

## How to Run the Pipeline

You can run the end-to-end training pipeline via the `train.py` script. The script handles loading data, engineering features, and running the specified model.

1. Navigate to the root directory of the project.
2. Open `src/train.py` and modify the `MODEL_TO_RUN` variable to choose which model you want to train (e.g., `'svr'` or `'random_forest'`).
3. Run the script:
   ```bash
   python src/train.py
   ```
