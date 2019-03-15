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
# random forest

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=40)

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

plt.figure('Prediksi', figsize=(4,4))
plt.imshow(xtes[0].reshape(64,64), cmap = 'gray')
plt.title('Aktual: {} / Prediksi: {} / Akurasi: {}'.format(
    ytes[0],
    model.predict([xtes[0]])[0],
    str(model.score(xtes, ytes) * 100) + ' %'
))

plt.show()