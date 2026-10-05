import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from collections import defaultdict
import numpy as np

# Load the training dataset
train_data = pd.read_csv('sign_mnist_13bal_train.csv')

# Separate the data (features) and the classes
X_train_full = train_data.drop('class', axis=1)  # Features (all columns except the first one)
X_train_full = X_train_full / 255.
y_train_full = train_data['class']   # Target (first column)

# Load the testing dataset
test_data = pd.read_csv('sign_mnist_13bal_test.csv')

# Separate the data (features) and the classes
X_test = test_data.drop('class', axis=1)  # Features (all columns except the first one)
X_test = X_test / 255.0 
y_test = test_data['class']   # Target (first column)

# Split training data into training and validation sets
X_train, X_validate, y_train, y_validate = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)

print(f"Training set size: {len(y_train)}")
print(f"Validation set size: {len(y_validate)}")
print(f"Test set size: {len(y_test)}")

# Create neural network model with a larger hidden layer
neural_net_model = MLPClassifier(
    hidden_layer_sizes=(64), #Try multiples of 8? !!64!!, 96
    random_state=40, 
    tol=0.005,
    max_iter=1000
)

# Train the model
neural_net_model.fit(X_train, y_train)

# Determine model architecture 
layer_sizes = [neural_net_model.coefs_[0].shape[0]]  # Start with the input layer size
layer_sizes += [coef.shape[1] for coef in neural_net_model.coefs_]  # Add sizes of subsequent layers
layer_size_str = " x ".join(map(str, layer_sizes))
print(f"Neural Network Architecture: {layer_size_str}")

# Make predictions on all datasets
y_pred_train = neural_net_model.predict(X_train)
y_pred_validate = neural_net_model.predict(X_validate)
y_pred_test = neural_net_model.predict(X_test)

# Calculate accuracies
train_accuracy = np.mean(y_pred_train == y_train) * 100
validation_accuracy = np.mean(y_pred_validate == y_validate) * 100
test_accuracy = np.mean(y_pred_test == y_test) * 100

print("----------")

print(f"Training Accuracy: {train_accuracy:.1f}%")
print(f"Validation Accuracy: {validation_accuracy:.1f}%")
print(f"Test Accuracy: {test_accuracy:.1f}%")

# Create dictionaries to hold total and correct counts for each class on test set
correct_counts = defaultdict(int)
total_counts = defaultdict(int)

# Count correct test predictions for each class
for true, pred in zip(y_test, y_pred_test):
    total_counts[true] += 1
    if true == pred:
        correct_counts[true] += 1

# Calculate and print accuracy
print("----------")

print("Accuracy per class: ")
for class_id in sorted(total_counts.keys()):
    accuracy = correct_counts[class_id] / total_counts[class_id] * 100
    print(f"Class {class_id}: {accuracy:3.0f}%")

conf_matrix = confusion_matrix(y_test, y_pred_test)
class_labels = sorted(list(set(y_test)))

misidentification_counts = {}
for i, true_class in enumerate(class_labels):
    for j, pred_class in enumerate(class_labels):
        if i != j and conf_matrix[i][j] > 0:
            misidentification_counts[(true_class, pred_class)] = conf_matrix[i][j]

top_misidentifications = sorted(misidentification_counts.items(), 
                              key=lambda x: x[1], reverse=True)[:3]

print("----------")

print("Commonly misidentified classes: ")
for i, ((true_class, pred_class), count) in enumerate(top_misidentifications, 1):
    print(f"{i}. Class {true_class} as Class {pred_class} -> {count} times")

class_accuracies = []
for class_id in sorted(total_counts.keys()):
    accuracy = correct_counts[class_id] / total_counts[class_id] * 100
    class_accuracies.append((class_id, accuracy))

low_classes = sorted(class_accuracies, key=lambda x: x[1])[:3]

print("----------")

print("Least accurate classes: ")
for i, (class_id, accuracy) in enumerate(low_classes, 1):
    print(f"{i}. Class {class_id}: {accuracy:.1f}% accuracy")
