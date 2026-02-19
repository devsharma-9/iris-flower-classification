from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from model import train_knn, train_decision_tree, evaluate_model

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train models
knn_model = train_knn(X_train, y_train)
dt_model = train_decision_tree(X_train, y_train)

# Evaluate models
print("\nKNN Model Performance:")
evaluate_model(knn_model, X_test, y_test)

print("\nDecision Tree Model Performance:")
evaluate_model(dt_model, X_test, y_test)
