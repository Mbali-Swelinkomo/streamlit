import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset into a pandas DataFrame
data = pd.read_csv('/home/mbali/Documents/Eduvos/ITDAA/heart.csv', delimiter=';')

# Handle Missing Values
missing_values = data.isnull().sum()
print("Missing Values:\n", missing_values)

# No missing values found

# Convert Categorical Variables (if any)

# No categorical variables found in the provided data.

# Normalize/Standardize Numerical Variables
scaler = StandardScaler()
numerical_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
                  'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# Feature Selection (if necessary)

# No feature selection performed in this example.

# Split Data
X = data.drop('target', axis=1)
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the preprocessed data
print("Preprocessed Data:")
print(X_train.head())

# Print column names
print(data.columns)