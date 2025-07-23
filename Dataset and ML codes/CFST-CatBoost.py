# Import necessary libraries
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score  # For model evaluation metrics
from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets
from catboost import CatBoostClassifier  # For using the CatBoost classifier
from sklearn.model_selection import KFold, cross_validate  # For cross-validation
from sklearn.preprocessing import MinMaxScaler  # For data normalization
from hyperopt import fmin, tpe, hp  # For hyperparameter optimization
import matplotlib.pyplot as plt  # For plotting
import warnings  # For handling warnings
warnings.filterwarnings(action='ignore')  # Ignore warnings to keep the output clean
import os  # For file and directory operations

# Set the working directory to the script's directory
script_dir = os.path.dirname(__file__)  # Get the directory of the script
os.chdir(script_dir)  # Change the working directory to the script's directory

# Load the data
df = pd.read_excel('dataset of rectangular CFST column.xlsx', sheet_name='data')  # Load the Excel file
data = df.values  # Extract the data as a NumPy array
X, y = data[:, :-1], data[:, -1]  # Split into features (X) and target (y)

# Split the dataset into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=1)  # Split into training and temporary sets
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=1)  # Split the temporary set into validation and testing sets

# Import the imbalanced-learn library for handling imbalanced datasets
import imblearn
from imblearn.combine import SMOTETomek  # For SMOTE-Tomek oversampling
smotetomek = SMOTETomek(random_state=1)  # Initialize the SMOTE-Tomek object
X_train, y_train = smotetomek.fit_resample(X_train, y_train)  # Apply SMOTE-Tomek to the training data

# Normalize the data
scaler = MinMaxScaler()  # Initialize the MinMaxScaler
X_train = scaler.fit_transform(X_train)  # Normalize the training data
X_val = scaler.transform(X_val)  # Normalize the validation data
X_test = scaler.transform(X_test)  # Normalize the testing data

# Define the hyperparameter search space
space = {
    'iterations': hp.quniform('iterations', 10, 300, 1),  # Number of boosting iterations
    'learning_rate': hp.loguniform('learning_rate', -2, -1),  # Learning rate
    'depth': hp.quniform('depth', 3, 16, 1),  # Maximum depth of trees
    'l2_leaf_reg': hp.loguniform('l2_leaf_reg', 1e-6, 10),  # L2 regularization on leaf values
    'random_strength': hp.uniform('random_strength', 0, 1),  # Random strength
    'border_count': hp.quniform('border_count', 1, 255, 1),  # Number of splits for numerical features
    'eval_metric': 'Accuracy'  # Evaluation metric
}

# Define the objective function for hyperparameter optimization
def objective(params):
    params['iterations'] = int(params['iterations'])  # Convert to integer
    params['depth'] = int(params['depth'])  # Convert to integer
    params['border_count'] = int(params['border_count'])  # Convert to integer
    final_model = CatBoostClassifier(**params, verbose=0)  # Initialize the CatBoost classifier
    final_model.fit(X_train, y_train)  # Train the model
    cv = KFold(n_splits=10, shuffle=True, random_state=1)  # 10-fold cross-validation
    validation_acc = cross_validate(final_model, X_val, y_val, scoring='accuracy', cv=cv, n_jobs=-1)  # Cross-validation accuracy
    return -np.mean(validation_acc['test_score'])  # Return the negative mean validation score

# Perform hyperparameter optimization
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=5, rstate=np.random.default_rng(1))  # Optimize using Hyperopt

# Extract the best hyperparameters
best_params = {
    'iterations': int(best['iterations']),  # Number of boosting iterations
    'learning_rate': best['learning_rate'],  # Learning rate
    'depth': int(best['depth']),  # Maximum depth of trees
    'l2_leaf_reg': best['l2_leaf_reg'],  # L2 regularization on leaf values
    'random_strength': best['random_strength'],  # Random strength
    'border_count': int(best['border_count']),  # Number of splits for numerical features
    'eval_metric': 'Accuracy'  # Evaluation metric
}
print("Best hyperparameters:", best_params)  # Print the best hyperparameters

# Train the final model with the best hyperparameters
final_model = CatBoostClassifier(**best_params, verbose=0)  # Initialize the CatBoost classifier with the best parameters
final_model.fit(X_train, y_train)  # Train the model

# Make predictions
y_pred = final_model.predict(X_test)  # Predict on the test set
y_train_pred = final_model.predict(X_train)  # Predict on the training set

# Print model evaluation metrics for the training set
print('**************************Model Evaluation Metrics for the Training Set*******************************')
print(f'Accuracy score: {round(accuracy_score(y_train, y_train_pred), 4)}')  # Accuracy
precision = precision_score(y_train, y_train_pred, average='micro')  # Precision
print(f'Precision: {round(precision, 4)}')  # Precision
recall = recall_score(y_train, y_train_pred, average='micro')  # Recall
print(f'Recall: {round(recall, 4)}')  # Recall
f1 = f1_score(y_train, y_train_pred, average='micro')  # F1 score
print(f'F1 score: {round(f1, 4)}')  # F1 score

# Print model evaluation metrics for the test set
print('**************************Model Evaluation Metrics for the Test Set*******************************')
print(f'Accuracy score: {round(accuracy_score(y_test, y_pred), 4)}')  # Accuracy
precision = precision_score(y_test, y_pred, average='micro')  # Precision
print(f'Precision: {round(precision, 4)}')  # Precision
recall = recall_score(y_test, y_pred, average='micro')  # Recall
print(f'Recall: {round(recall, 4)}')  # Recall
f1 = f1_score(y_test, y_pred, average='micro')  # F1 score
print(f'F1 score: {round(f1, 4)}')  # F1 score

# Print classification reports
from sklearn.metrics import classification_report  # For classification reports
print("Test set classification report:", classification_report(y_test, y_pred))  # Classification report for the test set
print("Training set classification report:", classification_report(y_train, y_train_pred))  # Classification report for the training set

# Plot and print confusion matrices
from sklearn.metrics import confusion_matrix  # For confusion matrices
import seaborn as sns  # For statistical data visualization

# Confusion matrix for the test set
cm_test = confusion_matrix(y_test, y_pred)  # Compute the confusion matrix
sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues", cbar=False)  # Plot the confusion matrix
plt.xlabel('Predicted Labels')  # X-axis label
plt.ylabel('True Labels')  # Y-axis label
plt.title('Test Set Confusion Matrix')  # Title
plt.show()  # Display the plot
print("cm_test=", cm_test)  # Print the confusion matrix

# Confusion matrix for the training set
cm_train = confusion_matrix(y_train, y_train_pred)  # Compute the confusion matrix
sns.heatmap(cm_train, annot=True, fmt="d", cmap="Blues", cbar=False)  # Plot the confusion matrix
plt.xlabel('Predicted Labels')  # X-axis label
plt.ylabel('True Labels')  # Y-axis label
plt.title('Training Set Confusion Matrix')  # Title
plt.show()  # Display the plot
print("cm_train=", cm_train)  # Print the confusion matrix
