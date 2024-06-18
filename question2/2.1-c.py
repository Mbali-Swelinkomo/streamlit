import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('../heart.csv', delimiter=';')

# Define numerical columns
numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'target']

# Create a single figure to display all plots
plt.figure(figsize=(16, 12))

# Plot distribution of classes for each numerical variable based on the target variable
for i, col in enumerate(numerical_cols[:-1], 1):
    plt.subplot(2, 3, i)
    sns.boxplot(x='target', y=col, data=data)
    plt.title(f"Distribution of {col} based on Target")
    plt.xlabel("Target")
    plt.ylabel(col)
    plt.xticks([0, 1], ['No Disease', 'Disease'])

# Adjust layout
plt.tight_layout()
plt.show()

# Observations that can be derived from these plots:
#
# Age: Generally, individuals with heart disease tend to be older compared to those without heart disease. Resting
# Blood Pressure (trestbps): There might not be a significant difference in resting blood pressure between
# individuals with and without heart disease. Cholesterol (chol): The distribution of cholesterol levels seems to be
# slightly higher in individuals with heart disease compared to those without. Maximum Heart Rate Achieved (thalach):
# Individuals without heart disease seem to have higher maximum heart rates compared to those with heart disease. ST
# Depression Induced by Exercise Relative to Rest (oldpeak): The ST depression seems to be higher in individuals with
# heart disease compared to those without.
