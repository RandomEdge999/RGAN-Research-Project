
import pandas as pd
import argparse
import os

def load_and_process_m5(data_dir, output_file, aggregate_level='total'):
    """
    Loads M5 dataset and processes it into a format suitable for RGAN.
    
    Args:
        data_dir: Directory containing 'sales_train_evaluation.csv' and 'calendar.csv'.
        output_file: Path to save the processed CSV.
        aggregate_level: Level of aggregation ('total' supported for now).
    """
    print(f"Loading data from {data_dir}...")
    
    try:
        calendar_path = os.path.join(data_dir, 'calendar.csv')
        sales_path = os.path.join(data_dir, 'sales_train_evaluation.csv')
        
        calendar = pd.read_csv(calendar_path)
        sales = pd.read_csv(sales_path)
        
        print("Data loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return False

    # Create a map from d_1, d_2, ... to date
    d_cols = [c for c in sales.columns if c.startswith('d_')]
    
    # Filter calendar to only include necessary days
    calendar = calendar[calendar['d'].isin(d_cols)]
    d_to_date = dict(zip(calendar['d'], calendar['date']))
    
    print(f"Processing {len(d_cols)} days of data...")

    if aggregate_level == 'total':
        print("Aggregating to total sales...")
        # Sum all sales for each day
        total_sales = sales[d_cols].sum(axis=0)
        
        # Create DataFrame
        df_output = pd.DataFrame({
            'date': [d_to_date[d] for d in total_sales.index],
            'sales': total_sales.values
        })
        
        # Ensure date is sorted
        df_output['date'] = pd.to_datetime(df_output['date'])
        df_output = df_output.sort_values('date')
        
        print(f"Saving processed data to {output_file}...")
        df_output.to_csv(output_file, index=False)
        print("Done!")

    else:
        print(f"Aggregation level '{aggregate_level}' not yet implemented.")

def cli_main():
    import sys
    parser = argparse.ArgumentParser(description="Process M5 dataset for RGAN.")
    parser.add_argument("--data_dir", type=str, default="data/m5", help="Directory containing M5 data files.")
    parser.add_argument("--output", type=str, default="m5_total.csv", help="Output CSV file.")
    args = parser.parse_args()
    result = load_and_process_m5(args.data_dir, args.output)
    if result is False:
        sys.exit(1)


if __name__ == "__main__":
    cli_main()
