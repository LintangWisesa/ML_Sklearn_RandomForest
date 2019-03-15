import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

digits = load_digits()
print(dir(digits))
# print(len(digits['data']))
# print(digits['images'][0])
# print(digits['target'][0])

# =============================
# create dataframe

df = pd.DataFrame(digits['data'])
df['target'] = digits['target']
# print(df.head(1))

from sklearn.model_selection import train_test_split
xtra, xtes, ytra, ytes = train_test_split(
    df.drop(['target'], axis = 'columns'), 
    df['target'], 
    test_size = .1
)
print(len(xtra))
print(len(xtes))

# =============================
# Random Forest

from sklearn.ensemble import RandomForestClassifier
modelR = RandomForestClassifier(n_estimators = 40)

# train
modelR.fit(xtra, ytra)

# accuracy
print(modelR.score(xtra, ytra) * 100, '%')

# prediction
# print(xtes.iloc[0])
print(modelR.predict([xtes.iloc[0]])[0])
print(ytes.iloc[0])

# =============================
# plot

xplot = xtes.iloc[0].values.reshape(8,8)
# print(xplot)

plt.figure('Digits', figsize=(5,5))
plt.imshow(xplot, cmap='gray')
plt.title('Aktual: {} / Prediksi: {} / Akurasi: {}'.format(
    ytes.iloc[0],
    modelR.predict([xtes.iloc[0]])[0],
    str(round(modelR.score(xtes, ytes) * 100)) + ' %'
))
plt.show()
