🧠 Logistic Regression in Python – Classification Techniques & Evaluation
This repository explores Logistic Regression, one of the most widely used classification algorithms in machine learning. Through hands-on Python implementations, we apply logistic regression in various real-world scenarios including binary, multiclass, and imbalanced data classifications.

🗂️ Contents
📘 Introduction to Logistic Regression

📊 Dataset Handling & Preprocessing

🧪 Model Training and Evaluation

🔍 Hyperparameter Tuning

🔄 Cross-Validation

⚖️ Handling Imbalanced Data

📈 Visualization & Metrics

💾 Model Saving & Loading

📌 Key Learning Outcomes
Apply logistic regression for binary and multiclass classification

Perform regularization: L1 (Lasso), L2 (Ridge), and ElasticNet

Evaluate models using accuracy, precision, recall, F1, ROC-AUC, MCC, Cohen’s Kappa

Implement hyperparameter tuning with GridSearchCV and RandomizedSearchCV

Use feature scaling and analyze its impact

Visualize confusion matrix and PR curve

Handle imbalanced datasets with class weights

Save and load models using joblib

🔧 Python Practice Tasks Included
🔹 Basic Logistic Regression
✅ Train-Test Split & Accuracy
Write a Python program that loads a dataset, splits it into training and testing sets, applies Logistic Regression, and prints the model accuracy.

✅ Load from CSV & Evaluate
Write a Python program to load a dataset from a CSV file, apply Logistic Regression, and evaluate its accuracy.

🔹 Regularization Techniques
🔒 L1 Regularization (Lasso)
LogisticRegression(penalty='l1', solver='liblinear')

🔐 L2 Regularization (Ridge)
LogisticRegression(penalty='l2')
→ Print model coefficients & accuracy

🔁 Elastic Net Regularization
LogisticRegression(penalty='elasticnet', l1_ratio=0.5, solver='saga')

🔹 Multiclass Classification
🎯 One-vs-Rest (OvR)
LogisticRegression(multi_class='ovr')

⚔️ One-vs-One (OvO)
Using OneVsOneClassifier(LogisticRegression()) from sklearn.multiclass

🔹 Hyperparameter Tuning
🔍 GridSearchCV for Logistic Regression
Tune C and penalty, print best parameters and accuracy.

🎲 RandomizedSearchCV for tuning C, penalty, and solver
→ Save computation time vs exhaustive grid search

🔹 Model Evaluation Techniques
🔄 Stratified K-Fold Cross-Validation
→ Print average accuracy across folds

📉 Precision, Recall, F1-Score Evaluation

📊 Confusion Matrix Visualization
→ Use seaborn.heatmap or ConfusionMatrixDisplay

📈 ROC-AUC Score

📏 Cohen’s Kappa Score

📊 Matthews Correlation Coefficient (MCC)

🔹 Advanced Use Cases
⚖️ Class Weights for Imbalanced Data
LogisticRegression(class_weight='balanced')

📏 Custom Regularization Strength (C=0.5)
→ Observe model flexibility and accuracy

🧮 Feature Importance from Coefficients
→ Identify which variables influence the outcome most

🧪 Compare Solvers (liblinear, saga, lbfgs)
→ Analyze solver impact on model performance

🔹 Data Engineering
🚢 Titanic Dataset Case Study

Handle missing values

Encode categorical features

Apply logistic regression and evaluate performance

⚙️ Feature Scaling (StandardScaler)

Train models on raw and standardized data

Compare accuracy

💾 Model Persistence
💽 Save and Load Model using joblib

Save trained model to .pkl

Load model and make new predictions

🛠️ Tools & Libraries
pandas, numpy – Data manipulation

scikit-learn – Modeling, evaluation, tuning

matplotlib, seaborn – Visualization

joblib – Model serialization

imbalanced-learn – Optional for imbalanced datasets
