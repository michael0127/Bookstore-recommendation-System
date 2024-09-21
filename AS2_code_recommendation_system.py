import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error

import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

# Read CSV files
merged = pd.read_csv("Final_datas.csv")
pd.set_option('display.max_columns', None)

# Useful columns
useful_columns = ['Year-Of-Publication',
   'Year-Of-Publication encoded', 'Book-Author Encoded',
   'Book-Publisher Encoded', 'User-State Encoded',
   'User-Age Encoded', 'User-Age', 'Book-Rating']

# Columns to be scaled
scale_columns = ['Year-Of-Publication',
   'Year-Of-Publication encoded', 'Book-Author Encoded',
   'Book-Publisher Encoded', 'User-State Encoded',
   'User-Age Encoded', 'User-Age',]

scaler = StandardScaler()

# Create a copy of merged samples to scale it
merged_data_scaled = merged[useful_columns].copy()

# Apply scaler on the sample data
merged_data_scaled[scale_columns] = scaler.fit_transform((merged_data_scaled[scale_columns]))


# Plot a correlation heatmapCheck to check correlation of features using heatmap
def corr_heatmap():
   corr_df = merged_data_scaled.corr()
   plt.figure()
   sns.heatmap(corr_df, annot=True, fmt='.2f')
   plt.title('Feature Correlation Heatmap')
   plt.show()


# Regressor
# Extract sample
sample = merged_data_scaled.sample(n=1000, random_state = 1)

# Define feature variables X and target variable y
X = sample.drop(columns=['Book-Rating'])
y = sample['Book-Rating']

# Experiment with different test sizes and cv values
test_size_list = [0.2, 0.25, 0.3]
cv_value_list = [5, 10]

X_encoded = sample[['Year-Of-Publication', 'Book-Author Encoded', 'Book-Publisher Encoded', 'User-State Encoded', 'User-Age Encoded']]

def visualize_decision_tree(feature=X_encoded, target=y, max_depth=3, random_state=1, figsize=(20, 10), fontsize=10):
   """
   Visualize a decision tree regressor.

   Parameters:
       feature (DataFrame, default=X_encoded): Feature data.
       target (Series, default=y): Target data.
       max_depth (int, default=3): Maximum depth of the decision tree.
       random_state (int, default=1): Random state for reproducibility.
       figsize (tuple, default=(20, 10)): Size of the figure.
       fontsize (int, default=10): Font size for the plot.
   """

   # Initialize the Decision Tree Regressor
   tree_model = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)

   # Fit the decision tree model
   tree_model.fit(feature, target)

   # Plot the decision tree
   plt.figure(figsize=figsize)
   plot_tree(tree_model, filled=True, feature_names=feature.columns, fontsize=fontsize)
   plt.title('Decision Tree Visualization')
   plt.show()


def feature_omission_impact(feature=X, target=y, random_state=1, test_size=0.2, n_neighbors=5):
   """
       Evaluate the impact of omitting each feature on the KNN model performance.

       Parameters:
       - feature: DataFrame, optional, default is X
           Features used for training the model.
       - target: Series, optional, default is y
           Target variable used for training the model.
       - random_state: int, optional, default is 1
           Random seed for reproducibility.
       - test_size: float, optional, default is 0.2
           Proportion of the dataset to include in the test split.
       - n_neighbors: int, optional, default is 5
           Number of neighbors to use in KNN.

   """

   # Split the data into training and testing sets
   X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=test_size, random_state=random_state)

   # Train the model with all features
   knn = KNeighborsRegressor(n_neighbors=n_neighbors)
   knn.fit(X_train, y_train)
   y_pred = knn.predict(X_test)
   baseline_mse = mean_squared_error(y_test, y_pred)

   # List to hold the increases in MSE
   mse_increases = []

   # Evaluate the impact of omitting each feature
   for feature in X.columns:
      # Drop the feature
      X_train_reduced = X_train.drop(columns=[feature])
      X_test_reduced = X_test.drop(columns=[feature])

      # Train a new model without this feature
      knn.fit(X_train_reduced, y_train)
      y_pred_reduced = knn.predict(X_test_reduced)
      reduced_mse = mean_squared_error(y_test, y_pred_reduced)

      # Calculate the increase in MSE
      mse_increase = reduced_mse - baseline_mse
      mse_increases.append(mse_increase)

   # Plot the results
   plt.figure(figsize=(10, 6))
   plt.bar(X.columns, mse_increases, color='skyblue')
   plt.xlabel('Features')
   plt.ylabel('Increase in MSE')
   plt.title('Impact of Omitting Each Feature on KNN Model Performance')
   plt.xticks(rotation=45)
   plt.tight_layout()
   plt.axhline(0, color='red', linestyle='--')  # Add a line at zero for reference
   plt.show()


def regressor_cross_val(regressor_type, feature=X, target=y, test_sizes=test_size_list, cv_values=cv_value_list,
                        random_state=1):
   """
   Evaluate regression models using cross-validation.

   Parameters:
       regressor_type (str): Type of regression model. 'Decision_Tree' or 'Linear'.
       feature (array-like, default=X): Feature data.
       target (array-like, default=y): Target data.
       test_sizes (list, default=test_size_list): List of test sizes for train-test split.
       cv_values (list, default=cv_value_list): List of cross-validation values.
       random_state (int, default=1): Random state for reproducibility.
   """

   # Initialize an empty list to store results
   results = []

   # Select the regressor based on the specified type
   if regressor_type == 'Decision_Tree':
      regressor = DecisionTreeRegressor(random_state=random_state)
   elif regressor_type == 'Linear':
      regressor = LinearRegression()
   else:
      raise ValueError("Unsupported classifier type. Use 'Decision_Tree' or 'Linear'.")

   # Iterate over test sizes
   for test_size in test_sizes:
      # Iterate over cross-validation values
      for cv_value in cv_values:
         # Split the dataset into training and testing sets
         X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=test_size)

         # Perform cross-validation for RMSE
         rmse_scores = cross_val_score(regressor, X_train, y_train, cv=cv_value,
                                       scoring='neg_root_mean_squared_error')

         # Perform cross-validation for R²
         r2_scores = cross_val_score(regressor, X_train, y_train, cv=cv_value, scoring='r2')

         # Store results
         results.append({'Test Size': test_size, 'CV': cv_value, 'RMSE': -np.mean(rmse_scores),
                         'R² Score': np.mean(r2_scores)})

   # Convert results to DataFrame
   results_df = pd.DataFrame(results)

   # Plotting
   plt.figure(figsize=(14, 6))

   # RMSE plot
   plt.subplot(1, 2, 1)
   sns.barplot(data=results_df, x='Test Size', y='RMSE', hue='CV')
   plt.title(f'RMSE for Different Test Sizes and CV - {regressor_type} Regression')
   plt.xlabel('Test Size')
   plt.ylabel('RMSE')
   plt.legend(title='CV')

   # R² Score plot
   plt.subplot(1, 2, 2)
   sns.barplot(data=results_df, x='Test Size', y='R² Score', hue='CV')
   plt.title(f'R² Score for Different Test Sizes and CV - {regressor_type} Regression')
   plt.xlabel('Test Size')
   plt.ylabel('R² Score')
   plt.legend(title='CV')

   plt.tight_layout()
   plt.show()

# Classifier
# Evaluates a classifier using cross-validation, classification report and confusion matrix
def evaluate_classifier(classifier_type, feature=X, target=y, test_size=0.3, cv_value=5, random_state=1):
   """
   Evaluate a classifier using cross-validation and test set.
   Parameters:
    - classifier_type: str
        Type of classifier to be evaluated. Supported types are 'Decision_Tree' and 'KNN'.
    - feature: DataFrame, optional, default is X
        Features used for training and testing the classifier.
    - target: Series, optional, default is y
        Target variable used for training and testing the classifier.
    - test_size: float, optional, default is 0.3
        Proportion of the dataset to include in the test split.
    - cv_value: int, optional, default is 5
        Number of folds in cross-validation.
    - random_state: int, optional, default is 1
        Random seed for reproducibility.
   """

   # Initialize the classifier based on the specified type
   if classifier_type == 'Decision_Tree':
      classifier = DecisionTreeClassifier(random_state=random_state)
   elif classifier_type == 'KNN':
      classifier = KNeighborsClassifier()
   else:
      raise ValueError("Unsupported classifier type. Use 'Decision_Tree' or 'KNN'.")

   # Split the dataset into training and testing sets
   X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=test_size, random_state=random_state)

   # Perform cross-validation
   accuracy_scores = cross_val_score(classifier, X_train, y_train, cv=cv_value, scoring='accuracy')
   f1_scores = cross_val_score(classifier, X_train, y_train, cv=cv_value, scoring='f1_macro')

   # Train the classifier on the full training set and evaluate on the test set
   classifier.fit(X_train, y_train)
   predictions = classifier.predict(X_test)

   # Generate confusion matrix
   confusion_mtx = confusion_matrix(y_test, predictions)

   # Print evaluation metrics
   print(f"Classifier: {classifier_type}, Test Size: {test_size}, CV: {cv_value}")
   print(f"Mean Accuracy: {np.mean(accuracy_scores)}")
   print(f"Mean F1-score: {np.mean(f1_scores)}")
   print("Classification Report:")
   print(classification_report(y_test, predictions))

   # Plot confusion matrix
   sns.heatmap(confusion_mtx, annot=True)
   plt.title(f'{classifier_type} Confusion Matrix')
   plt.xlabel("Predicted Value")
   plt.ylabel('Actual Value')
   plt.show()




corr_heatmap()
feature_omission_impact()

regressor_cross_val('Linear')

regressor_cross_val('Decision_Tree')
visualize_decision_tree()

evaluate_classifier('Decision_Tree')
evaluate_classifier('KNN')
