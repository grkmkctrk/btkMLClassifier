#################################################################
####################### Data Preprocesing #######################

"""**Kutuphaneler**"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""**Veri yukleme**"""

veriler = pd.read_csv('/content/eksikveriler.csv')

"""**DATA FRAME (SLICE) DILIMLEME**"""

X = veriler.iloc[:, [1]]
Y = veriler.iloc[:, [2]]
# NUMPY ARRAY DIZI DONUSUMU
x = X.values
y = Y.values

"""**Eksik verileri duzenleme**"""

from sklearn.impute import SimpleImputer

imp = SimpleImputer(missing_values=np.nan, strategy='mean')

sayisalVeriler = veriler.iloc[:, 1:4].values

sayisalVeriler = imp.fit_transform(sayisalVeriler)

sayisalVeriler

ulke = veriler.iloc[:,0:1].values

ulke

"""**Kategorikden sayisal veriye gecis**"""

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

ulke[:, 0] = le.fit_transform(veriler.iloc[:, 0:1])

ulke

ohe = preprocessing.OneHotEncoder()

ulke = ohe.fit_transform(ulke).toarray()

ulke

"""**Numpy dizisi DataFrame Donusumu**"""

sonuc1 = pd.DataFrame(data=ulke, index=range(len(ulke)), columns=['fr', 'tr', 'us'])

sonuc1

sonuc2 = pd.DataFrame(data=sayisalVeriler, index=range(len(sayisalVeriler)), columns= ['boy', 'kilo', 'yas'])

sonuc2


"""**DataFramelerin birlestirlime islemi**"""

s1 = pd.concat([sonuc1, sonuc2], axis=1) # satir satira birlestirmek icin yani yatay boyutta birlestirmek icin axis = 1

s2 = pd.concat([s1, sonuc3], axis=1)

s2

cinsiyet = veriler.iloc[:,-1].values

sonuc3 = pd.DataFrame(data=cinsiyet, index=range(len(cinsiyet)), columns=['cinsiyet'])

sonuc3

"""**Verilerin test ve egitim icin bolunmesi**"""

from sklearn.model_selection import train_test_split

# verinin yuzde 66 si antrenman icin kullanilsin kalan yuzde 33'u test edilsin diye ayrdik
# random_state rastsal ayirma icin kullaniliyor ayni degeri alan her kod ayni ayrimi yapar
x_train, x_test, y_train, y_test = train_test_split(s1, sonuc3, test_size = 0.33, random_state = 0)

"""**Verilerin normalize edilme islemi**"""

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

# burasi bir tur normalizasyon islemi anlatiyor (x-mean)/standar_deviation denklemini kullanir
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

####################### Data Preprocesing #######################
#################################################################

#################################################################
##################### Prediction Algorithms #####################
"""**LINEER REGRESSION**"""

X = veriler.iloc[:, 2:3]
Y = veriler.iloc[:, 5:]
# NUMPY ARRAY DIZI DONUSUMU
x = X.values
y = Y.values

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(x, y)

import statsmodels.api as sm
model1 = sm.OLS(lin_reg.predict(x),x).fit()
model1.summary()

"""**POLYNOMIAL REGRESSION**"""

X = veriler.iloc[:, 2:3]
Y = veriler.iloc[:, 5:]
# NUMPY ARRAY DIZI DONUSUMU
x = X.values
y = Y.values

from sklearn.preprocessing import PolynomialFeatures

# verileri derecesine gore olusturduk
poly_reg = PolynomialFeatures(degree = 2)

x_poly = poly_reg.fit_transform(x)

# polynomial predict
lin_regP = LinearRegression()
lin_regP.fit(x_poly, Y) # x_poly derecesine gore yeni X leri tanimlayan dizi

model2 = sm.OLS(lin_regP.predict(x_poly),x).fit()
model2.summary()

"""**SUPPORT VECTOR MACHINE**"""

X = veriler.iloc[:, 2:5]
Y = veriler.iloc[:, 5:]
# NUMPY ARRAY DIZI DONUSUMU
x = X.values
y = Y.values

from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(X)
sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(Y)

from sklearn.svm import SVR

svrReg = SVR(kernel = 'rbf')
svrReg.fit(x_olcekli, y_olcekli)

model3 = sm.OLS(svrReg.predict(x_olcekli),x_olcekli).fit()
model3.summary()

"""**DECISION TREE**"""

X = veriler.iloc[:, 2:3]
Y = veriler.iloc[:, 5:]
# NUMPY ARRAY DIZI DONUSUMU
x = X.values
y = Y.values

from sklearn.tree import DecisionTreeRegressor

dT = DecisionTreeRegressor(random_state = 0)
dT.fit(x,y)

model4 = sm.OLS(dT.predict(x),x).fit()
model4.summary()

"""**RANDOM FOREST**"""

X = veriler.iloc[:, 2:3]
Y = veriler.iloc[:, 5:]
# NUMPY ARRAY DIZI DONUSUMU
x = X.values
y = Y.values

from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(random_state = 0, n_estimators=10)
rfr.fit(x, y)

model5 = sm.OLS(rfr.predict(x),x).fit()
model5.summary()

print(f'LINEER REGRESSION       R-squared:   {model1.rsquared}')
print(f'POLYNOMIAL REGRESSION   R-squared:   {model2.rsquared}')
print(f'SUPPORT VECTOR MACHINE  R-squared:   {model3.rsquared}')
print(f'DECISION TREE           R-squared:   {model4.rsquared}')
print(f'RANDOM FOREST           R-squared:   {model5.rsquared}')

##################### Prediction Algorithms #####################
#################################################################

#################################################################
############## Evaluating of Method of Predictions ##############

from sklearn.metrics import r2_score
r2_score(y, y_pred)

############## Evaluating of Method of Predictions ##############
#################################################################
#-------------------------------------------------------------------------------

#################################################################
#################### Clasification Algorithms ###################

"""**Logistic Regression**"""
from sklearn.linear_model import LogisticRegression

logReg = LogisticRegression(random_state=0)

logReg.fit(X_train, y_train)

y_pred = logReg.predict(X_test)

"""**K Nearest Neighborhood Classification**"""
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1, metric='minkowski')

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

"""**SVM Classification**"""
from sklearn.svm import SVC

svc = SVC(kernel = 'rbf')
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)

"""**Naive Bayes Classification**"""

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)

mnb = MultinomialNB() # negatif data geldigi icin normalize edilmemis degerler ile kullandim
mnb.fit(x_train, y_train)

bnb = BernoulliNB()
bnb.fit(X_train, y_train)

y_predg = gnb.predict(X_test)
y_predm = mnb.predict(x_test)
y_predb = bnb.predict(X_test)

#################### Clasification Algorithms ###################
#################################################################

#################################################################
############ Evaluating of Method of Clasifications #############

"""**Confision Matrix**"""
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
cm

############ Evaluating of Method of Clasifications #############
#################################################################

