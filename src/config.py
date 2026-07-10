import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'Hourly_Data_220kV.csv')

SPLIT_DATE_TRAIN_END = '2025-04-01'
SPLIT_DATE_VAL_END = '2025-08-01'
