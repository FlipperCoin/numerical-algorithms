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

imagesPerDigit1 = images[:, :, labels == digit1]
imagesPerDigit2 = images[:, :, labels == digit2]

A_all = np.zeros((2*N, 28**2))
b_all = np.zeros((2*N, 1))

for i in range(N):
    A_all[2*i] = np.reshape(imagesPerDigit1[:, :, i], (28**2,))
    A_all[2*i+1] = np.reshape(imagesPerDigit2[:, :, i], (28**2,))
    b_all[2*i] = 1
    b_all[2*i+1] = -1

A_all = np.column_stack((A_all, np.ones(2*N)))

A_train = A_all[:N, :]
b_train = b_all[:N, :]
A_test = A_all[N:2*N, :]
b_test = b_all[N:2*N, :]

# Solve LS with train
x = pinv(A_train)@b_train

predC = np.sign(A_train @ x)
trueC = b_train
acc = np.mean(predC == trueC) * 100
print(f'Accuracy={acc}%, ({np.around((1 - acc / 100) * N)} wrong examples)')

predC = np.sign(A_test @ x)
trueC = b_test
acc = np.mean(predC == trueC) * 100
print(f'Accuracy={acc}%, ({np.around((1 - acc / 100) * N)} wrong examples)')

# plt.imshow(imagesPerDigit2[:,:,60], cmap='Greys')
# plt.show()