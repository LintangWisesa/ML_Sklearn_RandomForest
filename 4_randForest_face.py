import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_olivetti_faces
wajah = fetch_olivetti_faces()
print(dir(wajah))
# print(len(wajah['data'][0]))    # 64x64 = 4096
# print(wajah['images'][0])       # 64 arr @64
# print(wajah['target'])          # 40 person @10 pose

# ===============================
# split: 90% train & 10% test

from sklearn.model_selection import train_test_split
xtra, xtes, ytra, ytes = train_test_split(
    wajah['data'], 
    wajah['target'], 
    test_size = .1
)
# print(len(xtra))
# print(len(xtes))
# print(xtra[0])
# print(ytra[0])

# ===============================
# decision tree

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()

# train
model.fit(xtra, ytra)

# akurasi
print(model.score(xtes, ytes))

# predict
print(xtes[0])
print(model.predict([xtes[0]]))
print(ytes[0])

# ===============================
# plot

# plt.figure('Wajah', figsize=(10,4))
# for i in range(10):
    # person = 15       
    # start: 0 end: 39
    # plt.subplot(2, 5, i + 1)
    # plt.imshow(wajah['images'][i + (10 * person)], cmap = 'gray')
    # plt.suptitle('Wajah orang ke-{}'.format(person))

plt.figure('Prediksi', figsize=(4,4))
plt.imshow(xtes[0].reshape(64,64), cmap = 'gray')
plt.title('Aktual: {} / Prediksi: {}'.format(
    ytes[0],
    model.predict([xtes[0]])[0]
))

plt.show()