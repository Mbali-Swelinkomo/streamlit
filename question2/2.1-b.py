import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('../heart.csv', delimiter=';')

# Define categorical columns
categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'target']

# Create a single figure to display all plots
plt.figure(figsize=(16, 12))

# Plot distribution of classes for each categorical variable based on the target variable
for i, col in enumerate(categorical_cols, 1):
    plt.subplot(3, 3, i)
    sns.countplot(x=col, hue='target', data=data)
    plt.title(f"Distribution of {col} based on Target")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.legend(title='Target', labels=['No Disease', 'Disease'])

# Adjust layout
plt.tight_layout()
plt.show()

# Observation: more males have the disease
