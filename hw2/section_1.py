import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from numpy.linalg import pinv

N = 4000
digit1 = 0
digit2 = 7

data = loadmat('mnist.mat')['training']
labels = np.array([x[0] for x in data['labels'][0][0]])

images = data['images'][0][0]

imagesPerDigit = [images[:, :, labels == 0], images[:, :, labels == 1], images[:, :, labels == 2], images[:, :, labels == 3], images[:, :, labels == 4], images[:, :, labels == 5], images[:, :, labels == 6], images[:, :, labels == 7], images[:, :, labels == 8], images[:, :, labels == 9]]

imagesPerDigit1 = imagesPerDigit[0]

imagesPerDigit2 = np.zeros((28, 28, N))
for i in range(N):
    imagesPerDigit2[:, :, i] = imagesPerDigit[(i % 9) + 1][:, :, i // 9]

A_all = np.zeros((2 * N, 28 ** 2))
b_all = np.zeros((2 * N, 1))

for i in range(N):
    A_all[2 * i] = np.reshape(imagesPerDigit1[:, :, i], (28 ** 2,))
    A_all[2 * i + 1] = np.reshape(imagesPerDigit2[:, :, i], (28 ** 2,))
    b_all[2 * i] = 1
    b_all[2 * i + 1] = -1

A_all = np.column_stack((A_all, np.ones(2 * N)))

A_train = A_all[:N, :]
b_train = b_all[:N, :]
A_test = A_all[N:2 * N, :]
b_test = b_all[N:2 * N, :]

# Solve LS with train
x = pinv(A_train) @ b_train

predC = np.sign(A_train @ x)
trueC = b_train
acc = np.mean(predC == trueC) * 100
print(f'Accuracy={acc}%, ({(1 - acc / 100) * N} wrong examples)')

predC = np.sign(A_test @ x)
trueC = b_test
acc = np.mean(predC == trueC) * 100
print(f'Accuracy={acc}%, ({(1 - acc / 100) * N} wrong examples)')
#
# plt.imshow((A_test[529, :-1]).reshape(28, 28), cmap='Greys')
# plt.show()
