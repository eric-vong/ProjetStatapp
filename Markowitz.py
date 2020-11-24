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
portfolio_norisk_estimated = np.matmul(inv_corr_estimated,e)/ (np.matmul(np.matmul(np.transpose(e),inv_corr_estimated),e))
portfolio_risk_estimated = np.matmul(inv_corr_estimated,mean_estimated) -np.matmul(np.matmul(np.transpose(e),inv_corr_estimated),mean_vect)*np.matmul(inv_corr_estimated,e) / np.matmul(np.matmul(np.transpose(e),inv_corr_estimated),e)

inv_corr = np.linalg.inv(corr_matrix)
portfolio_norisk = np.matmul(inv_corr,e)/ (np.matmul(np.matmul(np.transpose(e),inv_corr),e))
portfolio_risk = np.matmul(inv_corr,mean_vect) - (np.matmul(np.matmul(np.transpose(e),inv_corr),mean_vect))*np.matmul(inv_corr,e) / np.matmul(np.matmul(np.transpose(e),inv_corr),e)

def markowitz_real_sol(lambda_m):
    return (portfolio_norisk + portfolio_risk/lambda_m)

def markowitz_estimated_sol(lambda_m):
    return (portfolio_norisk_estimated + portfolio_risk_estimated/lambda_m)

def efficient_frontier_real(lambda_sample=100):
    R_real = np.zeros(lambda_sample)
    sigma_real = np.zeros(lambda_sample)
    for k in np.arange(0,lambda_sample):
        R_real[k] = np.matmul(np.transpose(mean_vect),markowitz_real_sol(k))
        sigma_real[k] = np.matmul(np.matmul(np.transpose(markowitz_real_sol(k)),corr_matrix),markowitz_real_sol(k))
    return R_real,sigma_real

def efficient_frontier_estimated_in_reality(lambda_sample=100):
    R_estimated_real = np.zeros(lambda_sample)
    sigma_estimated_real = np.zeros(lambda_sample)
    for k in np.arange(0,lambda_sample):
        R_estimated_real[k] = np.matmul(np.transpose(mean_vect),markowitz_estimated_sol(k))
        sigma_estimated_real[k] = np.matmul(np.matmul(np.transpose(markowitz_estimated_sol(k)),corr_matrix),markowitz_estimated_sol(k))
    return R_estimated_real,sigma_estimated_real
    
def efficient_frontier_estimated(lambda_sample=100):
    R_estimated = np.zeros(lambda_sample)
    sigma_estimated = np.zeros(lambda_sample)
    for k in np.arange(0,lambda_sample):
        R_estimated[k] = np.matmul(np.transpose(mean_estimated),markowitz_estimated_sol(k))
        sigma_estimated[k] = np.matmul(np.matmul(np.transpose(markowitz_estimated_sol(k)),corr_matrix_estimated),markowitz_estimated_sol(k))
    return R_estimated,sigma_estimated

def monte_carlo_estimator(mean_vector,corr_matrix,size, M = 10000):
    R_monte_carlo = np.zeros(size)
    sigma_monte_carlo = np.zeros(size)
    for m in np.arange(M) :
        R,sigma = efficient_frontier_estimated_in_reality(size)
        R_monte_carlo += R
        sigma_monte_carlo += sigma
    return R_monte_carlo/M,sigma_monte_carlo/M

R_real,sigma_real = efficient_frontier_real(100)
R_estimated,sigma_estimated = efficient_frontier_estimated(100)
R_estimated_real,sigma_estimated_real = efficient_frontier_estimated_in_reality(100)
R_monte_carlo,sigma_monte_carlo = monte_carlo_estimator(mean_vect,corr_matrix,1024)

eigenvalues_real = np.sort(np.linalg.eigvals(corr_matrix))
eigenvalues_estimated = np.sort(np.linalg.eigvals(corr_matrix_estimated))

deviation = (eigenvalues_real - eigenvalues_estimated)/eigenvalues_real * 100
print(deviation)

plt.plot(sigma_real,R_real,'b') 
plt.plot(sigma_estimated,R_estimated,'g')
plt.plot(sigma_estimated_real,R_estimated_real,'r')
plt.plot(sigma_monte_carlo,R_monte_carlo,'y')
plt.show()