import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

n = 39

df_m = pd.read_csv(r'F:\Desktop\Projet_Statapp\data\mean_vector.csv',sep=';')
df_corr = pd.read_csv(r'F:\Desktop\Projet_Statapp\data\covariance_matrix.csv',sep=';')

corr_matrix = np.zeros((n,n))
mean_vect = np.zeros(n)
e = np.ones(n)

for i in range(n):
    mean_vect[i] = float(df_m.iloc[0,i])
    for j in range(n):
        corr_matrix[i,j] = float(df_corr.iloc[i,j+1])

inv_corr = np.linalg.inv(corr_matrix)
portfolio_norisk = (inv_corr * e)/ (np.transpose(e)*inv_corr*e)
portfolio_risk = inv_corr * mean_vect - np.transpose(e)*inv_corr*mean_vect*inv_corr*e / (np.transpose(e)*inv_corr*e)

def markowitz_solution(lambda):
    return portfolio_norisk + portfolio_risk/lambda

lambda_space = [10**i for i in np.arange(0,100)]
markowitz_sol = [markowitz_solution(lambda) for lambda in lambda_space]

plt.plot(lambda_space,markowitz_sol)

