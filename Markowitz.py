import numpy as np
import pandas as pd 
import matplotlib as plt

n = 39

df_m = pd.read_csv(r'F:\Desktop\Projet_Statapp\CAC40_Moyenne.csv',sep=';')
df_corr = pd.read_csv(r'F:\Desktop\Projet_Statapp\CAC40_Covariance.csv',sep=';')

corr_matrix = np.zeros((n,n))
mean_vect = np.zeros(n)
e = np.ones(n)

for i in range(n):
    mean_vect[i] = float(df_m.iloc[0,i])
    for j in range(n):
        corr_matrix[i,j] = float(df_corr.iloc[i,j+1])

inv_corr = np.linalg.inv(corr_matrix)