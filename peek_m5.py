
import pandas as pd

try:
    print("Reading calendar.csv...")
    calendar = pd.read_csv('m5-forecasting-accuracy/calendar.csv', nrows=5)
    print(calendar.head())
    
    print("\nReading sales_train_evaluation.csv...")
    sales = pd.read_csv('m5-forecasting-accuracy/sales_train_evaluation.csv', nrows=5)
    print(sales.head())
except Exception as e:
    print(f"Error: {e}")
