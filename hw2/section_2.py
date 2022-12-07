import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from numpy.linalg import pinv

N = 800
data = loadmat('mnist.mat')['training']
labels = np.array([x[0] for x in data['labels'][0][0]])
images = data['images'][0][0]
imagesPerDigit = [images[:, :, labels == 0], images[:, :, labels == 1], images[:, :, labels == 2],
                  images[:, :, labels == 3], images[:, :, labels == 4], images[:, :, labels == 5],
                  images[:, :, labels == 6], images[:, :, labels == 7], images[:, :, labels == 8],
                  images[:, :, labels == 9]]

A_all = np.zeros((10 * N, 28 ** 2))
b_all = np.zeros((10 * N, 1))

for i in range(N):
    for j in range(10):
        A_all[10 * i + j] = np.reshape(imagesPerDigit[j][:, :, i], (28 ** 2,))

A_all = np.column_stack((A_all, np.ones(10 * N)))
A_train = A_all[:5 * N, :]
A_test = A_all[5 * N:10 * N, :]

def train_weights(k):
    for i in range(N):
        for j in range(10):
            b_all[10 * i + j] = 1 if j == k else -1

    b_train = b_all[:5 * N, :]
    b_test = b_all[5 * N:10 * N, :]

    # Solve LS with train
    x = pinv(A_train) @ b_train

    return x.reshape((785,)), b_train.reshape((5*N,)), b_test.reshape((5*N,))

x = np.zeros((28**2+1, 10))
b_train = np.zeros((5*N, 10))
b_test = np.zeros((5*N, 10))
for k in range(10):
    x[:, k], b_train[:, k], b_test[:, k] = train_weights(k)

predC_train = np.sign(A_train @ x)
predC_test = np.sign(A_test @ x)

for k in range(10):
    acc = np.mean(predC_train[:, k] == b_train[:, k]) * 100
    print(f'Classifier for {k}, train: accuracy={acc}%, ({np.around((1 - acc / 100) * 5*N)} wrong examples)')

    acc = np.mean(predC_test[:, k] == b_test[:, k]) * 100
    print(f'Classifier for {k}, test: accuracy={acc}%, ({np.around((1 - acc / 100) * 5*N)} wrong examples)')

    wrong = predC_test != b_test
    for i in np.argwhere(wrong)[:1]:
        plt.imshow((A_test[i[0], :-1]).reshape(28, 28), cmap='Greys')
        plt.axis('off')
        plt.savefig(f"sec2_{k}_{i[0]}.png")
        plt.show()

