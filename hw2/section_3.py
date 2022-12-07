import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from numpy.linalg import pinv

N = 800
data_all = loadmat('mnist.mat')
data = data_all['training']
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
for k in range(10):
    x[:, k], b_train, b_test = train_weights(k)

data = data_all['test']
true_labels = np.array([x[0] for x in data['labels'][0][0]])
images = data['images'][0][0]
N = images.shape[2]
A_all = np.zeros((N, 28 ** 2))
for i in range(N):
    A_all[i] = np.reshape(images[:, :, i], (28**2,))

A_all = np.column_stack((A_all, np.ones(N)))

predC = A_all @ x
labels = np.zeros(N, dtype=int)
uncertain_count = 0
for i in range(len(predC)):
    if np.all(predC[i] < 0):
        uncertain_count += 1
        diff = np.abs(predC[i] + 1)
        closest = np.argmax(diff)
    else:
        if np.sum(predC[i] > 0) > 1:
            uncertain_count += 1
        diff = np.abs(predC[i] - 1)
        closest = np.argmin(diff)

    labels[i] = closest

acc = np.mean(labels == true_labels) * 100
print(f'Accuracy={acc}%, ({np.around((1 - acc / 100) * N)} wrong examples), uncertain answers = {uncertain_count}')

wrong = labels != true_labels
for i in np.argwhere(wrong)[40:45]:
    plt.imshow((A_all[i[0], :-1]).reshape(28, 28), cmap='Greys')
    plt.axis('off')
    plt.savefig(f"sec3_{i[0]}_classified_{labels[i[0]]}_true_{true_labels[i[0]]}.png")
    plt.show()
