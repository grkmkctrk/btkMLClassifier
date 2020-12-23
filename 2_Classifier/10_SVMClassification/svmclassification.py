# -*- coding: utf-8 -*-
"""SVMClassification.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1JO5zfjqW2qJVyUCtZxaqy7u7xESVd9Fj
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler = pd.read_csv('/content/veriler.csv')

X = veriler.iloc[:, 1:4]
Y = veriler.iloc[:, 4:] # dependent
# NUMPY ARRAY DIZI DONUSUMU
x = X.values
y = Y.values

from sklearn.model_selection import train_test_split

# verinin yuzde 66 si antrenman icin kullanilsin kalan yuzde 33'u test edilsin diye ayrdik
# random_state rastsal ayirma icin kullaniliyor ayni degeri alan her kod ayni ayrimi yapar
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

# burasi bir tur normalizasyon islemi anlatiyor (x-mean)/standar_deviation denklemini kullanir

X_train = sc.fit_transform(x_train)   # ogren ve ogrendiklerine gore donustur
X_test = sc.transform(x_test)         # daha once ogrendigin gibi donustur

from sklearn.svm import SVC

svc = SVC(kernel = 'rbf')
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
# satirlar y_test  | sutunlar y_pred oldu
#       e k
#     e 1 0
#     k 1 6

cm

