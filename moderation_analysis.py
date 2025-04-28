import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('Merged_Urban_and_Rural_Dataset.csv', encoding='cp1252')

# Print initial data info
print("\nInitial data info:")
print(f"Total rows: {len(df)}")
print("\nColumns in dataset:")
print(df.columns.tolist())

# Check for missing values in key columns
key_columns = [
    'SES',
    'Do you attend any regular extracurricular activities (clubs, sections) in addition to school lessons and classes in school subjects? (Single choice)',
    'SocialMots_Cooperation',
    'Well-being_Cooperation'
]

print("\nMissing values in key columns:")
for col in key_columns:
    if col in df.columns:
        missing = df[col].isnull().sum()
        print(f"{col}: {missing} missing values ({missing/len(df)*100:.1f}%)")
        print(f"Unique values: {df[col].unique()}")
    else:
        print(f"{col}: Column not found")

# Prepare the data
print("\nPreparing data...")

# 1. Handle SES variable
if 'SES' in df.columns:
    print("\nSES values before mapping:")
    print(df['SES'].value_counts())
    
    # Map SES values
    ses_mapping = {
        'Low SES': 'SESLow',
        'Middle SES': 'SESMiddle',
        'High SES': 'SESHigh'
    }
    df['SES'] = df['SES'].map(ses_mapping)
    df['SES_Category'] = pd.Categorical(df['SES'], 
                                      categories=['SESLow', 'SESMiddle', 'SESHigh'],
                                      ordered=True)
    
    print("\nSES values after mapping:")
    print(df['SES_Category'].value_counts())

# 2. Handle extracurricular participation
extracurricular_col = 'Do you attend any regular extracurricular activities (clubs, sections) in addition to school lessons and classes in school subjects? (Single choice)'
if extracurricular_col in df.columns:
    print("\nExtracurricular values before mapping:")
    print(df[extracurricular_col].value_counts())
    
    df['Extracurricular_Binary'] = df[extracurricular_col].map({'Yes': 1, 'No': 0})
    
    print("\nExtracurricular values after mapping:")
    print(df['Extracurricular_Binary'].value_counts())

# 3. Rename the Well-being_Cooperation column to remove hyphen
df = df.rename(columns={'Well-being_Cooperation': 'Wellbeing_Cooperation'})

# 4. Clean the data
print("\nData cleaning:")
print(f"Initial number of rows: {len(df)}")

# Check for missing values in all required columns
required_columns = ['SES_Category', 'Extracurricular_Binary', 'SocialMots_Cooperation', 'Wellbeing_Cooperation']
missing_in_columns = {col: df[col].isnull().sum() for col in required_columns if col in df.columns}
print("\nMissing values in required columns:")
for col, missing in missing_in_columns.items():
    print(f"{col}: {missing} missing values ({missing/len(df)*100:.1f}%)")

# Remove rows with missing values in key variables
df_clean = df.dropna(subset=required_columns)

print(f"\nNumber of rows after cleaning: {len(df_clean)}")
print(f"Number of rows removed: {len(df) - len(df_clean)}")

if len(df_clean) > 0:
    # Print sample sizes for each group
    print("\nSample sizes by group:")
    print(df_clean.groupby(['SES_Category', 'Extracurricular_Binary']).size())

    def analyze_moderation(outcome_var, data):
        print(f"\nAnalyzing {outcome_var}...")
        
        # Fit the model
        formula = f"{outcome_var} ~ C(SES_Category) + Extracurricular_Binary + C(SES_Category):Extracurricular_Binary"
        model = ols(formula, data=data).fit()
        
        # Print results
        print("\nRegression Results:")
        print("="*50)
        print(f"R-squared: {model.rsquared:.3f}")
        print(f"Adjusted R-squared: {model.rsquared_adj:.3f}")
        print(f"F-statistic: {model.fvalue:.3f}")
        print(f"P-value (F-statistic): {model.f_pvalue:.4f}")
        
        print("\nCoefficients and P-values:")
        results_df = pd.DataFrame({
            'Coefficient': model.params,
            'P-value': model.pvalues
        })
        print(results_df)
        
        # Create interaction plot
        plt.figure(figsize=(10, 6))
        sns.pointplot(x='SES_Category', y=outcome_var, hue='Extracurricular_Binary', 
                     data=data, dodge=True, markers=['o', 's'],
                     linestyles=['-', '--'], palette='Set2')
        plt.title(f'Interaction Plot: {outcome_var}')
        plt.xlabel('Socioeconomic Status')
        plt.ylabel(outcome_var)
        plt.legend(title='Extracurricular Participation', labels=['No', 'Yes'])
        plt.tight_layout()
        plt.savefig(f'{outcome_var}_interaction_plot.png')
        plt.close()
        
        return model

    # Run analyses for both outcomes
    print("\nRunning moderation analyses...")
    social_mots_model = analyze_moderation('SocialMots_Cooperation', df_clean)
    wellbeing_model = analyze_moderation('Wellbeing_Cooperation', df_clean)

    print("\nAnalysis complete!")
    print("Interaction plots have been saved as:")
    print("- SocialMots_Cooperation_interaction_plot.png")
    print("- Wellbeing_Cooperation_interaction_plot.png")
else:
    print("\nError: No data remaining after cleaning. Please check the data structure and missing values.") 