import pandas as pd
import numpy as np

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('Merged_Urban_and_Rural_Dataset.csv', encoding='latin1', low_memory=False)

# Print basic information
print("\nDataset Overview:")
print("-----------------")
print(f"Total number of rows: {len(df)}")
print(f"Total number of columns: {len(df.columns)}")

# Check required columns
required_columns = ['SES', 'Family_size', 'School Location', 
                   'SocialMots_Cooperation', 'Well-being_Cooperation']

print("\nRequired Columns Analysis:")
print("-------------------------")
for col in required_columns:
    print(f"\n{col}:")
    if col in df.columns:
        print("✓ Column exists")
        print(f"Data type: {df[col].dtype}")
        print(f"Missing values: {df[col].isnull().sum()} ({(df[col].isnull().sum()/len(df))*100:.2f}%)")
        
        if col == 'Family_size':
            print("\nFamily Size Categories:")
            print("BHF: Broken Home Family (one parent)")
            print("NFS: Nuclear Family with one child")
            print("NFC: Nuclear Family with more than one child")
            print("GF: Grand Family (living with grandparents)")
            print("\nDistribution of Family Types:")
            family_counts = df[col].value_counts()
            for family_type, count in family_counts.items():
                percentage = (count / len(df)) * 100
                print(f"{family_type}: {count} ({percentage:.1f}%)")
        else:
            print("\nUnique values:")
            print(df[col].value_counts().head())
    else:
        print("✗ Column not found in dataset")

# Check correlations between variables
print("\nCorrelation Analysis:")
print("--------------------")
# Convert Family_size to numeric for correlation analysis
family_mapping = {
    'BHF': 1,
    'NFS': 2,
    'NFC': 3,
    'GF': 4
}
df['Family_size_numeric'] = df['Family_size'].map(family_mapping)

numeric_cols = ['Family_size_numeric', 'SocialMots_Cooperation', 'Well-being_Cooperation']
if all(col in df.columns for col in numeric_cols):
    correlations = df[numeric_cols].corr()
    print("\nCorrelations between numeric variables:")
    print(correlations)

# Print sample rows
print("\nSample Data:")
print("------------")
print(df[required_columns].head())

# Check for potential data quality issues
print("\nPotential Data Quality Issues:")
print("-----------------------------")
for col in required_columns:
    if col in df.columns:
        if col == 'Family_size':
            # Check for unexpected family types
            valid_family_types = ['BHF', 'NFS', 'NFC', 'GF']
            invalid_types = df[~df[col].isin(valid_family_types)][col].unique()
            if len(invalid_types) > 0:
                print(f"\n{col} has unexpected values:")
                print(invalid_types)
        elif df[col].dtype == 'object':
            print(f"\n{col} unique values:")
            print(df[col].unique()) 