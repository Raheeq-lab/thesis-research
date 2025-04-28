import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import sys

def check_numeric(df, columns):
    """Check if columns are numeric and convert if possible"""
    for col in columns:
        if not pd.to_numeric(df[col], errors='coerce').notnull().all():
            print(f"\nWarning: Non-numeric values found in {col}")
            print("Sample of non-numeric values:")
            non_numeric = df[pd.to_numeric(df[col], errors='coerce').isnull()]
            print(non_numeric[col].head())
            return False
    return True

def validate_data(df):
    """Validate the dataset for required columns and data types"""
    required_columns = ['SES', 'Family_size', 'School Location', 
                       'SocialMots_Cooperation', 'Well-being_Cooperation']
    
    # Check if all required columns exist
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Print initial sample size
    print(f"\nInitial sample size: {len(df)}")
    
    # Check for missing values
    missing_values = df[required_columns].isnull().sum()
    if missing_values.any():
        print("\nMissing values in required columns:")
        print(missing_values[missing_values > 0])
        # Remove rows with missing values
        df = df.dropna(subset=required_columns)
        print(f"Sample size after removing missing values: {len(df)}")
    
    # Check data types and unique values
    print("\nChecking column data types and unique values:")
    for col in required_columns:
        print(f"\n{col}:")
        print(f"Data type: {df[col].dtype}")
        print("Sample unique values:", df[col].unique()[:5])
    
    return df

def create_regression_plots(df, results):
    """Create additional visualizations for regression results"""
    
    # 1. Coefficient Plot
    plt.figure(figsize=(12, 6))
    for var in ['SocialMots_Cooperation', 'Well-being_Cooperation']:
        models = results[var]
        coefs = []
        predictors = []
        for i, model in enumerate(models, 1):
            if i == 1:
                coefs.extend(model.params[1:])  # Exclude intercept
                predictors.extend(['SES'])
            elif i == 2:
                coefs.extend(model.params[1:])
                predictors.extend(['SES', 'Family_size'])
            else:
                coefs.extend(model.params[1:])
                predictors.extend(['SES', 'Family_size', 'School Location'])
        
        plt.subplot(1, 2, 1 if var == 'SocialMots_Cooperation' else 2)
        plt.bar(predictors, coefs)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.title(f'Standardized Coefficients - {var}')
        plt.ylabel('Beta Coefficient')
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('regression_coefficients.png')
    plt.close()
    
    # 2. R-squared Comparison Plot
    plt.figure(figsize=(10, 6))
    r_squared = []
    models = ['SES', 'SES + Family', 'Full Model']
    for var in ['SocialMots_Cooperation', 'Well-being_Cooperation']:
        r_squared.append([model.rsquared for model in results[var]])
    
    x = np.arange(len(models))
    width = 0.35
    
    plt.bar(x - width/2, r_squared[0], width, label='SocialMots_Cooperation')
    plt.bar(x + width/2, r_squared[1], width, label='Well-being_Cooperation')
    
    plt.xlabel('Model')
    plt.ylabel('R-squared')
    plt.title('R-squared Comparison Across Models')
    plt.xticks(x, models)
    plt.legend()
    plt.tight_layout()
    plt.savefig('r_squared_comparison.png')
    plt.close()
    
    # 3. Interaction Plots
    plt.figure(figsize=(15, 5))
    
    # SES vs Cooperation by School Location
    plt.subplot(1, 3, 1)
    sns.boxplot(x='SES', y='SocialMots_Cooperation', hue='School Location', data=df)
    plt.title('SES vs SocialMots_Cooperation by School Location')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 3, 2)
    sns.boxplot(x='SES', y='Well-being_Cooperation', hue='School Location', data=df)
    plt.title('SES vs Well-being_Cooperation by School Location')
    plt.xticks(rotation=45)
    
    # Family Size vs Cooperation
    plt.subplot(1, 3, 3)
    sns.boxplot(x='Family_size', y='SocialMots_Cooperation', data=df)
    plt.title('Family Size vs SocialMots_Cooperation')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('interaction_plots.png')
    plt.close()
    
    # 4. Residual Plots
    plt.figure(figsize=(15, 5))
    
    for i, var in enumerate(['SocialMots_Cooperation', 'Well-being_Cooperation'], 1):
        model = results[var][2]  # Full model
        plt.subplot(1, 2, i)
        plt.scatter(model.fittedvalues, model.resid)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.xlabel('Fitted Values')
        plt.ylabel('Residuals')
        plt.title(f'Residual Plot - {var}')
    
    plt.tight_layout()
    plt.savefig('residual_plots.png')
    plt.close()

try:
    # Load the dataset
    print("Loading dataset...")
    df = pd.read_csv('Merged_Urban_and_Rural_Dataset.csv', encoding='latin1')
    
    # Validate data
    print("\nValidating data...")
    df = validate_data(df)
    
    # Convert School Location to dummy variable (0 = Rural, 1 = Urban)
    print("\nPreprocessing data...")
    print("Converting School Location to dummy variables...")
    df['School Location'] = (df['School Location'] == 'Urban').astype(int)
    
    # Convert SES to numerical values
    print("\nConverting SES to numerical values...")
    ses_mapping = {
        'Low SES': 1,
        'Middle SES': 2,
        'High SES': 3
    }
    df['SES'] = df['SES'].map(ses_mapping)
    
    # Convert Family_size to numerical values
    print("\nConverting Family_size to numerical values...")
    family_size_mapping = {
        'BHF': 1,  # Broken Home Family
        'NFS': 2,  # Nuclear Family Single child
        'NFC': 3,  # Nuclear Family Multiple children
        'GF': 4    # Grand Family
    }
    df['Family_size'] = df['Family_size'].map(family_size_mapping)
    
    # Remove rows with missing or invalid values
    initial_size = len(df)
    df = df.dropna(subset=['SES', 'Family_size', 'School Location', 
                          'SocialMots_Cooperation', 'Well-being_Cooperation'])
    if len(df) < initial_size:
        print(f"\nRemoved {initial_size - len(df)} rows with invalid values")
        print(f"Final sample size: {len(df)}")
    
    # Standardize continuous variables
    print("\nStandardizing variables...")
    scaler = StandardScaler()
    df[['SES', 'Family_size']] = scaler.fit_transform(df[['SES', 'Family_size']])
    
    # Define dependent variables
    dependent_vars = ['SocialMots_Cooperation', 'Well-being_Cooperation']
    
    # Create a function to run hierarchical regression
    def run_hierarchical_regression(dependent_var):
        # Step 1: SES only
        X1 = df[['SES']]
        X1 = sm.add_constant(X1)
        model1 = sm.OLS(df[dependent_var], X1).fit()
        
        # Step 2: SES + Family_size
        X2 = df[['SES', 'Family_size']]
        X2 = sm.add_constant(X2)
        model2 = sm.OLS(df[dependent_var], X2).fit()
        
        # Step 3: Full model (SES + Family_size + School Location)
        X3 = df[['SES', 'Family_size', 'School Location']]
        X3 = sm.add_constant(X3)
        model3 = sm.OLS(df[dependent_var], X3).fit()
        
        return model1, model2, model3
    
    # Run analysis for each dependent variable
    print("\nRunning regression analysis...")
    results = {}
    for var in dependent_vars:
        print(f"\nAnalyzing {var}...")
        results[var] = run_hierarchical_regression(var)
    
    # Save results to a text file
    print("\nSaving results...")
    with open('regression_results.txt', 'w', encoding='utf-8') as f:
        f.write("HIERARCHICAL MULTIPLE REGRESSION RESULTS\n")
        f.write("=======================================\n\n")
        f.write(f"Final sample size: {len(df)}\n\n")
        
        for var in dependent_vars:
            f.write(f"Analysis for {var}\n")
            f.write("-" * len(f"Analysis for {var}") + "\n\n")
            
            models = results[var]
            for i, model in enumerate(models, 1):
                f.write(f"Model {i}:\n")
                if i == 1:
                    f.write("Predictors: SES\n")
                elif i == 2:
                    f.write("Predictors: SES + Family_size\n")
                else:
                    f.write("Predictors: SES + Family_size + School Location\n")
                
                f.write(f"R-squared: {model.rsquared:.4f}\n")
                f.write(f"Adjusted R-squared: {model.rsquared_adj:.4f}\n")
                f.write(f"F-statistic: {model.fvalue:.4f}\n")
                f.write(f"F-statistic p-value: {model.f_pvalue:.4f}\n\n")
                
                f.write("Coefficients:\n")
                f.write("-------------\n")
                for name, coef, pval in zip(model.params.index, model.params, model.pvalues):
                    f.write(f"{name}: beta = {coef:.4f}, p = {pval:.4f}\n")
                f.write("\n" + "="*50 + "\n\n")
    
    # Create correlation matrix plot
    print("\nCreating correlation matrix plot...")
    plt.figure(figsize=(10, 8))
    corr_matrix = df[['SES', 'Family_size', 'School Location', 
                     'SocialMots_Cooperation', 'Well-being_Cooperation']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.close()
    
    # After saving results, create additional visualizations
    print("\nCreating additional visualizations...")
    create_regression_plots(df, results)
    
    print("\nAnalysis complete. Results saved to 'regression_results.txt' and visualizations saved as PNG files.")

except Exception as e:
    print(f"\nError occurred: {str(e)}")
    sys.exit(1) 