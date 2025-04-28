import pandas as pd
import sys

try:
    # Load the dataset with latin1 encoding
    print("Loading dataset...")
    df = pd.read_csv('Merged_Urban_and_Rural_Dataset.csv', encoding='latin1', low_memory=False)
    
    # Print all column names
    print("\nAll column names in the dataset:")
    for i, col in enumerate(df.columns):
        print(f"{i}: {col}")
    
    # Look specifically for school location related columns
    print("\nSearching for school location columns...")
    location_columns = []
    for col in df.columns:
        if any(term in col.lower() for term in ['school', 'location', 'urban', 'rural', 'area', 'type']):
            location_columns.append(col)
            print(f"\nFound possible location column: {col}")
            print(f"Sample values: {df[col].head()}")
            print(f"Unique values: {df[col].unique()}")
    
    if not location_columns:
        print("\nNo school location columns found!")
    else:
        print("\nPossible school location columns found:")
        for col in location_columns:
            print(f"- {col}")

except Exception as e:
    print(f"Error: {str(e)}")
    sys.exit(1) 