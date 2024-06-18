import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

data = pd.read_csv('/home/mbali/Documents/Eduvos/ITDAA/heart.csv', delimiter=';')


# Split the data into features (X) and target variable (y)
X = data.drop(columns=['target'])
y = data['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the machine learning models
models = [
    ("Random Forest", RandomForestClassifier()),
    ("Support Vector Machine", SVC()),
    ("K-Nearest Neighbors", KNeighborsClassifier())
]

# Train and evaluate each model
best_model = None
best_accuracy = 0.0
for name, model in models:
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy}")
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

# Save the best model to disk
joblib.dump(best_model, '../question4/heart_disease_model.pkl')
print("Best Model Saved!")


# For heart disease prediction, several machine learning models can be considered. Here are three appropriate models
# along with their advantages and disadvantages:
#
# Logistic Regression:
# Logistic Regression is a linear classification algorithm used when the target variable is binary. It models the
# probability that an instance belongs to a particular class.
# Advantages:
# Simple and easy to implement. Provides probabilities for outcomes.
# Disadvantages:
# Assumes linear relationship between features and target. May not perform well with non-linear relationships.
#
# Random Forest:
# Random Forest is an ensemble learning method that constructs a multitude of decision trees during training and outputs
# the mode of the classes as the prediction result.
# Advantages:
# Robust to overfitting due to averaging of multiple trees. Handles non-linear relationships well.
# Disadvantages:
# May be slow to predict due to the complexity of multiple trees. Not easily interpretable compared to
# simpler models like logistic regression.
#
# Gradient Boosting Classifier:
# Gradient Boosting builds an ensemble of decision trees sequentially, where each tree corrects the errors of the
# previous one. It combines the predictions from multiple weak learners to create a strong learner.
# Advantages:
# Often provides higher accuracy compared to other models. Handles complex relationships between features and target
# variable.
# Disadvantages:
# More prone to overfitting if hyperparameters are not tuned properly. Training can be computationally expensive.
