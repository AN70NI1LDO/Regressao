import numpy as np
import pandas as pd
from sklearn import linear_model

#lendo arquivo

adms = pd.read_csv ("/content/drive/MyDrive/datasetExerc/Admission_Predict_Ver1.1.csv")
adms.info()
adms.columns

#features e targets

X = adms [adms.columns [:-1]]
X

y = adms [adms.columns [-1:]]
y

# Dataframe

import matplotlib.pyplot as plt
for i in range (X.shape [1]):
  print (i)
  plt.scatter (X.iloc[:, i], y)
  plt.show()

# selecionando Dataframe

X2 = X.iloc [:, 1:2]
plt.scatter (X2, y)
plt.show()

# manipulando dados

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt

adms_lr = LinearRegression()

dlt = 1 # Delta
interp = adms_lr.intercept_ # Variavel W
iterc = 30 # Qtd iteracoes

lst_W = [] #intercept
lst_W = np.linspace (interp + dlt, interp - dlt, iterc)

lst_mse = []
for v in range (iter):
  adms_lr.intercept_ = lst_W [v]
  ypred = adms_lr.predict (X2)
  mse = mean_squared_error (y, ypred)
  lst_mse.append (mse)

print (lst_mse)

plt.plot (list_W, lst_mse)
plt.show()
