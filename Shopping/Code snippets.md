
# PCA
```python

import pandas as pd  
import numpy as np  
from sklearn.decomposition import PCA  
from sklearn.preprocessing import StandardScaler  
import matplotlib.pyplot as plt

  
# PCA: Perform PCA on the dataset to reduce dimensions focusing on Annual Income, Spending Score, Items Purchased (  
# Count), and Session Duration. Choose components based on explained variance and visualize the first two principal  
# components.  
  
  
# Selecting relevant features  
# Adjust these feature names based on the dataset you wish to apply PCA on later.  
features = ['Annual Income (k$)', 'Spending Score (1-100)', 'Items Purchased (Count)', 'Session Duration (min)']  
x = df.loc[:, features].values  
  
# Standardizing the features  
# It's important to standardize features before applying PCA since PCA is sensitive to variances of the initial variables.  
x = StandardScaler().fit_transform(x)  
  
# PCA execution  
pca = PCA(n_components=2)  # you can change the number of components for different datasets  
principal_components = pca.fit_transform(x)  
  
# Create a DataFrame with the principal components for easier access and further analysis  
principal_df = pd.DataFrame(data=principal_components, columns=['Principal Component 1', 'Principal Component 2'])  
  
# Explained variance can help determine how many components should be kept  
explained_variance = pca.explained_variance_ratio_  
print("Explained Variance: ", explained_variance)  
  
# Visualizing the first two principal components  
plt.figure(figsize=(8, 6))  
plt.scatter(principal_df['Principal Component 1'], principal_df['Principal Component 2'])  
plt.xlabel('Principal Component 1')  
plt.ylabel('Principal Component 2')  
plt.title('PCA of Customer Data')  
plt.grid(True)  
plt.show()

```


# Similarity and statistics

```python
# Calculate the Euclidean and cosine similarity between customers based on their # Spending Score and Items Purchased. Use summary statistics to derive probabilities of a customer returning items.
  
# Select the features 'Spending Score (1-100)' and 'Items Purchased (Count)'  
features = df[['Spending Score (1-100)', 'Items Purchased (Count)']].values  
  
# Calculate Euclidean distance between each pair of customers  
euclidean_dist = euclidean_distances(features)  
  
# Calculate Cosine similarity between each pair of customers  
cosine_sim = cosine_similarity(features)  
  
# Convert distance matrix to a DataFrame for better readability  
euclidean_df = pd.DataFrame(euclidean_dist, index=df['CustomerID'], columns=df['CustomerID'])  
cosine_sim_df = pd.DataFrame(cosine_sim, index=df['CustomerID'], columns=df['CustomerID'])  
  
# Summary statistics for 'Returned Items (Count)' to calculate the probability of returning items  
returned_items_stats = df['Returned Items (Count)'].describe()  
total_customers = returned_items_stats['count']  
customers_with_returns = df[df['Returned Items (Count)'] > 0].shape[0]  
  
# Probability of a customer returning items  
probability_of_return = customers_with_returns / total_customers  
  
print(f"Probability of a customer returning items: {probability_of_return:.2f}")


# Result: 
# Probability of a customer returning items: 0.79

```


## KDF and Histogram

```python
# Create visualizations for the distribution of Session Duration across different customer ages using histograms and  
# KDE plots. Interpret any patterns you observe.  
  
# Plotting histograms and KDEs for Session Duration across different age ranges  
# Setting up the aesthetic style of the plots  
sns.set(style="whitegrid")  
  
# Defining age bins for grouping customers  
age_bins = [18, 30, 40, 50, 60, 70]  
age_labels = ['18-29', '30-39', '40-49', '50-59', '60-69']  
df['Age Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)  
  
# Creating a figure and a set of subplots  
plt.figure(figsize=(14, 7))  
  
# Creating a histogram with KDE  
sns.histplot(df, x="Session Duration (min)", hue="Age Group", element="step", stat="density", common_norm=False,  
             kde=True)  
  
# Title and labels  
plt.title('Distribution of Session Duration Across Different Customer Ages')  
plt.xlabel('Session Duration (min)')  
plt.ylabel('Density')  
  
# Display the plot  
plt.show()
```


## Decision tree
```python
  
# Decision Tree Model to predict 'Category'  
  
# Define the features and the target variable for classification  
features = df.drop(columns=['CustomerID', 'Category'])  # drop non-numeric and target columns for features  
target = df['Category']  # target variable  
  
# Split the dataset into training and testing sets for the decision tree  
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2, random_state=42)  
  
# Initialize and fit the decision tree classifier  
decision_tree = DecisionTreeClassifier(random_state=42)  
decision_tree.fit(features_train, target_train)  
  
# Predict the 'Category' for the testing set  
target_pred = decision_tree.predict(features_test)  
  
# Calculate the accuracy of the model  
dt_accuracy = accuracy_score(target_test, target_pred)  
print(f"Decision Tree Model Accuracy: {dt_accuracy:.2f}")  
  
# Visualize the decision tree (for simplicity, only depth 3 is visualized)  
plt.figure(figsize=(20,10))  
tree.plot_tree(decision_tree, max_depth=3, feature_names=features.columns, class_names=decision_tree.classes_, filled=True)  
plt.show()  
  
# Linear Regression to model the relationship between 'Annual Income' and 'Spending Score'  
  
# Define the independent variable (Annual Income) and dependent variable (Spending Score)  
X = df['Annual Income (k$)'].values.reshape(-1,1)  # Independent variable  
y = df['Spending Score (1-100)'].values  # Dependent variable  
  
# Split the dataset into training and testing sets for the linear regression  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  
  
# Initialize and fit the linear regression model  
lin_reg = LinearRegression()  
lin_reg.fit(X_train, y_train)  
  
# Predict the 'Spending Score' for the testing set  
y_pred = lin_reg.predict(X_test)  
  
# Calculate the mean squared error of the model  
mse = mean_squared_error(y_test, y_pred)  
print(f"Linear Regression Model Mean Squared Error: {mse:.2f}")  
  
# Plotting the regression line against the data points  
plt.scatter(X_test, y_test, color='blue')  
plt.plot(X_test, y_pred, color='red', linewidth=2)  
plt.title('Annual Income vs Spending Score')  
plt.xlabel('Annual Income (k$)')  
plt.ylabel('Spending Score (1-100)')  
plt.show()
```