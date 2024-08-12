import numpy as np
import matplotlib.pyplot as plt
from helpers.sammon import sammon

G1 = np.load('Gram_cubic__full_FCM.npy')
G2 = np.load('Gram_cubic__full_DCM.npy')
G3 = np.load('Gram_cubic__full_PDM.npy')



sigma1, U1 = np.linalg.eigh(G1)
sigma2, U2 = np.linalg.eigh(G2)
sigma3, U3 = np.linalg.eigh(G3)

data1 = np.diag(sigma1[1:]) @ U1[:, 1:].T
data2 = np.diag(sigma2[1:]) @ U2[:, 1:].T
data3 = np.diag(sigma3[1:]) @ U3[:, 1:].T

# normalize columns of G1
for i, g in enumerate(data1):
    data1[i] = g / np.linalg.norm(g)
# # normalize columns of G2
# for i, g in enumerate(data2):
#     data2[i] = g / np.linalg.norm(g)
# # normalize columns of G3
# for i, g in enumerate(data3):
#     data3[i] = g / np.linalg.norm(g)

data = data1

x, _ = sammon(data.T, 2)
x = data.T

plt.scatter(x[:22, 0], x[:22, 1], c='#2ca02c') # HH: green
plt.scatter(x[22:44, 0], x[22:44, 1], c='#e377c2') # DD: pink
plt.scatter(x[44:, 0], x[44:, 1], c='#7f7f7f') # HD: grey
plt.show()