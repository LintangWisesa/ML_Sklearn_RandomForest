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
# plot

plt.figure('Digits', figsize=(10,4))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(digits['images'][i], cmap='gray')
plt.show()
