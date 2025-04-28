import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('Merged_Urban_and_Rural_Dataset.csv', encoding='latin1')

# Function to perform t-test and calculate effect size
def analyze_variable(variable_name, group_var='School Location'):
    # Remove missing values
    data = df[[group_var, variable_name]].dropna()
    
    urban = data[data[group_var] == 'Urban'][variable_name]
    rural = data[data[group_var] == 'Rural'][variable_name]
    
    # Calculate descriptive statistics
    urban_mean = urban.mean()
    rural_mean = rural.mean()
    urban_std = urban.std()
    rural_std = rural.std()
    
    # Perform t-test
    t_stat, p_value = stats.ttest_ind(urban, rural, equal_var=False)
    
    # Calculate Cohen's d
    n1, n2 = len(urban), len(rural)
    pooled_std = np.sqrt(((n1-1)*urban_std**2 + (n2-1)*rural_std**2) / (n1 + n2 - 2))
    cohen_d = (urban_mean - rural_mean) / pooled_std
    
    # Print results
    print(f"\nAnalysis for {variable_name}:")
    print("-" * (len(f"Analysis for {variable_name}:") + 1))
    print(f"Sample sizes:")
    print(f"  Urban: n={n1}")
    print(f"  Rural: n={n2}")
    print(f"\nDescriptive Statistics:")
    print(f"  Urban: Mean = {urban_mean:.3f}, SD = {urban_std:.3f}")
    print(f"  Rural: Mean = {rural_mean:.3f}, SD = {rural_std:.3f}")
    print(f"\nInferential Statistics:")
    print(f"  Independent samples t-test:")
    print(f"    t({n1+n2-2}) = {t_stat:.3f}")
    print(f"    p = {p_value:.4f}")
    print(f"    Cohen's d = {cohen_d:.3f}")
    
    # Interpret effect size
    if abs(cohen_d) < 0.2:
        effect_size = "negligible"
    elif abs(cohen_d) < 0.5:
        effect_size = "small"
    elif abs(cohen_d) < 0.8:
        effect_size = "medium"
    else:
        effect_size = "large"
    
    print(f"\nEffect size interpretation:")
    print(f"  Cohen's d = {cohen_d:.3f} indicates a {effect_size} effect")
    
    return urban, rural

print("\nPerforming statistical analyses...")
# Analyze both cooperation variables
social_mots_urban, social_mots_rural = analyze_variable('SocialMots_Cooperation')
wellbeing_urban, wellbeing_rural = analyze_variable('Well-being_Cooperation')

print("\nCreating visualizations...")
# Create boxplots
plt.figure(figsize=(15, 6))

# SocialMots_Cooperation boxplot
plt.subplot(1, 2, 1)
sns.boxplot(x='School Location', y='SocialMots_Cooperation', data=df)
plt.title('Social Motives Cooperation by School Location')
plt.ylabel('Social Motives Cooperation Score')

# Well-being_Cooperation boxplot
plt.subplot(1, 2, 2)
sns.boxplot(x='School Location', y='Well-being_Cooperation', data=df)
plt.title('Well-being Cooperation by School Location')
plt.ylabel('Well-being Cooperation Score')

plt.tight_layout()
plt.savefig('cooperation_boxplots.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nAnalysis complete. Boxplots have been saved as 'cooperation_boxplots.png'") 