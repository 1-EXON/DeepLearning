import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

X = np.array([ 1.0, 3.0 ])
W1 = np.array([ [1.0, 2.0, 3.0], [4.0, 5.0, 6.0] ])
B1 = 2.0
S1 = np.dot(X, W1) + B1
S1 = S1
print(S1, S1.shape)

W2 = np.array([ [1.0, 2.0], [3.0, 4.0], [5.0, 6.0] ])
B2 = 3.0
S2 = np.dot(S1, W2) + B2
S2 = sigmoid(S2)
print(S2, S2.shape)

W3 = np.array([ [1.0], [2.0] ])
B3 = 1.0
S3 = np.dot(S2, W3) + B1
print(S3, S3.shape)