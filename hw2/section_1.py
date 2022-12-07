import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from numpy.linalg import pinv

N = 800

data = loadmat('mnist.mat')['training']
labels = np.array([x[0] for x in data['labels'][0][0]])

images = data['images'][0][0]

imagesPerDigit = [images[:, :, labels == 0], images[:, :, labels == 1], images[:, :, labels == 2], images[:, :, labels == 3], images[:, :, labels == 4], images[:, :, labels == 5], images[:, :, labels == 6], images[:, :, labels == 7], images[:, :, labels == 8], images[:, :, labels == 9]]

A_all = np.zeros((10 * N, 28 ** 2))
b_all = np.zeros((10 * N, 1))

for i in range(N):
    for k in range(10):
        A_all[10 * i + k] = np.reshape(imagesPerDigit[k][:, :, i], (28 ** 2,))
        b_all[10 * i + k] = 1 if k == 0 else -1

A_all = np.column_stack((A_all, np.ones(10 * N)))

A_train = A_all[:5*N, :]
b_train = b_all[:5*N, :]
A_test = A_all[5*N:10 * N, :]
b_test = b_all[5*N:10 * N, :]

# Solve LS with train
x = pinv(A_train) @ b_train

predC = np.sign(A_train @ x)
trueC = b_train
acc = np.mean(predC == trueC) * 100
print(f'Train: accuracy={acc}%, ({np.around((1 - acc / 100) * 5*N)} wrong examples)')

predC = np.sign(A_test @ x)
trueC = b_test
acc = np.mean(predC == trueC) * 100
print(f'Test: accuracy={acc}%, ({np.around((1 - acc / 100) * 5*N)} wrong examples)')

wrong = predC != trueC
for i in np.argwhere(wrong)[:5]:
    plt.imshow((A_test[i[0], :-1]).reshape(28, 28), cmap='Greys')
    plt.axis('off')
    plt.savefig(f"sec1_{i[0]}.png")
    plt.show()

