import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

n = 39
sample_size = 250

df_m = pd.read_csv(r'F:\Desktop\Projet_Statapp\data\mean_vector.csv',sep=';')
df_corr = pd.read_csv(r'F:\Desktop\Projet_Statapp\data\covariance_matrix.csv',sep=';')

corr_matrix = np.zeros((n,n))
mean_vect = np.zeros(n)
e = np.ones(n)

for i in range(n):
    mean_vect[i] = float(df_m.iloc[0,i])
    for j in range(n):
        corr_matrix[i,j] = float(df_corr.iloc[i,j+1])

def sample_generation(mean_vect,corr_matrix,p = sample_size):
    return np.random.multivariate_normal(mean_vect,corr_matrix,sample_size)

def mean_estimation(sample,p=sample_size):
    mean_estimated = np.zeros(n)
    for i in range(p) :
        for j in range(n):
            mean_estimated[j]+= sample[i][j]
    mean_estimated = mean_estimated/p
    return mean_estimated

def corr_matrix_estimation(sample,mean_estimation,p = sample_size):
    sample_centered = sample - mean_estimation
    return np.matmul(np.transpose(sample_centered),sample_centered)/p

sample = sample_generation(mean_vect,corr_matrix)
mean_estimated = mean_estimation(sample)
corr_matrix_estimated = corr_matrix_estimation(sample,mean_estimated)

inv_corr_estimated = np.linalg.inv(corr_matrix_estimated)
portfolio_norisk_estimated = (inv_corr_estimated * e)/ (np.transpose(e)*inv_corr_estimated*e)
portfolio_risk_estimated = inv_corr_estimated * mean_estimated - np.transpose(e)*inv_corr_estimated*mean_estimated*inv_corr_estimated*e / (np.transpose(e)*inv_corr_estimated*e)

inv_corr = np.linalg.inv(corr_matrix)
portfolio_norisk = (inv_corr * e)/ (np.transpose(e)*inv_corr*e)
portfolio_risk = inv_corr * mean_vect - np.transpose(e)*inv_corr*mean_vect*inv_corr*e / (np.transpose(e)*inv_corr*e)

def markowitz_real_solution(lambda_m):
    return (portfolio_norisk + portfolio_risk/lambda_m)

def markowitz_estimated_solution(lambda_m):
    return (portfolio_norisk_estimated + portfolio_risk_estimated/lambda_m)

lambda_space = [10**i for i in np.arange(0,100)]
markowitz_real_sol = [markowitz_real_solution(lambda_m) for lambda_m in lambda_space]
markowitz_estimated_sol = [markowitz_estimated_solution(lambda_m) for lambda_m in lambda_space]

