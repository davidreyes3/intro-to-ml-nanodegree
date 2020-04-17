import numpy as np
import pandas as pd

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Output (prediction) formula
def output_formula(features, weights, bias):
    return sigmoid(np.dot(features, weights) + bias)

def gradient_descent():
    data = pd.read_csv('data.csv', header=None)
    features = np.array(data[[0, 1]])
    y = np.array(data[2])

    n_records, n_features = features.shape
    weights = np.random.normal(scale=1 / n_features ** .5, size=n_features)
    bias = 0
    
    print(np.dot(features, weights))

    out = output_formula(features, weights, bias)