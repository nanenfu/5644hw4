import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# 1. Data Generation
# Parameter Settings
r_minus_1, r_plus_1 = 2, 4
sigma = 1
n_train, n_test = 1000, 10000

# Function to Generate Data
def generate_data(n_samples, r_value):
    theta = np.random.uniform(-np.pi, np.pi, n_samples)  # Random angle
    n = np.random.normal(0, sigma, (n_samples, 2))  # Noise
    x = np.array([r_value * np.cos(theta), r_value * np.sin(theta)]).T + n
    return x

# Generate Training and Testing Data
x_train_class_minus_1 = generate_data(n_train // 2, r_minus_1)
x_train_class_plus_1 = generate_data(n_train // 2, r_plus_1)
x_test_class_minus_1 = generate_data(n_test // 2, r_minus_1)
x_test_class_plus_1 = generate_data(n_test // 2, r_plus_1)

# Combine Data and Generate Labels
x_train = np.vstack((x_train_class_minus_1, x_train_class_plus_1))
y_train = np.array([-1] * (n_train // 2) + [1] * (n_train // 2))
x_test = np.vstack((x_test_class_minus_1, x_test_class_plus_1))
y_test = np.array([-1] * (n_test // 2) + [1] * (n_test // 2))

# Data Standardization
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Data Visualization
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap='bwr', alpha=0.6)
plt.title('Training Data Distribution')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

# 2. Model Selection and Hyperparameter Optimization
# Define SVM Model and Hyperparameter Range
svm_params = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.1, 1, 10]
}
svm = SVC(kernel='rbf')

# Use GridSearchCV to Find Best Hyperparameters (10-Fold Cross Validation)
svm_grid = GridSearchCV(svm, svm_params, cv=10, return_train_score=True)
svm_grid.fit(x_train, y_train)
best_svm = svm_grid.best_estimator_
print(f'Best SVM Parameters: {svm_grid.best_params_}')

# Visualize SVM Hyperparameter Cross-Validation Results
svm_results = svm_grid.cv_results_
sns.heatmap(
    data=np.array(svm_results['mean_test_score']).reshape(len(svm_params['C']), len(svm_params['gamma'])),
    annot=True, cmap='viridis', xticklabels=svm_params['gamma'], yticklabels=svm_params['C']
)
plt.xlabel('Gamma')
plt.ylabel('C')
plt.title('SVM Cross-Validation Accuracy')
plt.show()

# Define MLP Model and Hyperparameter Range
mlp_params = {
    'hidden_layer_sizes': [(5,), (10,), (15,)],
    'activation': ['relu', 'tanh', 'logistic']
}
mlp = MLPClassifier(max_iter=2000, learning_rate_init=0.001, tol=1e-4)

# Use GridSearchCV to Find Best Hyperparameters (10-Fold Cross Validation)
mlp_grid = GridSearchCV(mlp, mlp_params, cv=10, return_train_score=True)
mlp_grid.fit(x_train, y_train)
best_mlp = mlp_grid.best_estimator_
print(f'Best MLP Parameters: {mlp_grid.best_params_}')

# Visualize MLP Hyperparameter Cross-Validation Results
mlp_results = mlp_grid.cv_results_
mlp_scores = np.array(mlp_results['mean_test_score']).reshape(len(mlp_params['hidden_layer_sizes']), len(mlp_params['activation']))
sns.heatmap(
    data=mlp_scores,
    annot=True, cmap='viridis', xticklabels=mlp_params['activation'], yticklabels=[str(h) for h in mlp_params['hidden_layer_sizes']]
)
plt.xlabel('Activation Function')
plt.ylabel('Hidden Layer Sizes')
plt.title('MLP Cross-Validation Accuracy')
plt.show()

# 3. Model Testing and Performance Evaluation
# Evaluate SVM and MLP Performance on Test Data
y_pred_svm = best_svm.predict(x_test)
y_pred_mlp = best_mlp.predict(x_test)

svm_accuracy = accuracy_score(y_test, y_pred_svm)
mlp_accuracy = accuracy_score(y_test, y_pred_mlp)

print(f'SVM Test Accuracy: {svm_accuracy:.4f}')
print(f'MLP Test Accuracy: {mlp_accuracy:.4f}')

# 4. Decision Boundary Visualization
def plot_decision_boundaries(models, titles, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    plt.figure(figsize=(12, 6))
    for i, (model, title) in enumerate(zip(models, titles), 1):
        plt.subplot(1, 2, i)
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.8, cmap='bwr')
        plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap='bwr', alpha=0.6)
        plt.title(title)
        plt.xlabel('x1')
        plt.ylabel('x2')
    plt.tight_layout()
    plt.show()

# Visualize SVM and MLP decision boundaries together
plot_decision_boundaries([best_svm, best_mlp], ['SVM Decision Boundary', 'MLP Decision Boundary'], x_test, y_test)

