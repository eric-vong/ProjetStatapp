import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from math import sqrt
import time

df = pd.read_csv(r'F:\Desktop\Projet_Statapp\data\CAC40.csv', sep=';',decimal=',')
start = '23-10-2010'
end = '23-10-2020'
df = df.drop(columns=['DATES','WLN FP Equity']) #On supprime les dates et la dernière colonne, entrée trop tardive
df = df.dropna() #On enlève toutes les lignes où il manque au moins une donnée
df = df.reset_index(drop=True) #On réordonne les indices, faire attention pas toujours bien si on veut calculer le return monthly
df = df.apply(lambda x: 100*x/x[0]) #On normalise 
returns_daily = df.apply(lambda x: (x/(x.shift(1))-1)) #Retours journaliers
returns_total = df.apply(lambda x: (x/x[0])-1) #Retour si on achète au début et on vend à la date t
returns_total = returns_total.iloc[1:]
returns_daily = returns_daily.iloc[1:] #On enlève la première ligne de NaN.
d = len(df.columns) #Nombres de colonnes
e = np.ones(d)

def mean_cov_dataframe(df): #On renvoie la moyenne + la covariance de la BDD
    mean = df.mean()
    cov = df.cov()
    return mean,cov

def sample_generation(mean,cov,sample_size = 250): #Génération de l'échantillon + calcul de sa moyenne + covariance (à changer selon estimateur)
    sample = np.random.multivariate_normal(mean,cov,sample_size)
    sample_mean_estim = sample.mean(0)
    sample_cov_estim = (sample - sample_mean_estim).T@(sample - sample_mean_estim)/(sample_size -1)
    return sample,sample_mean_estim,sample_cov_estim

def eigvalues(mean,cov,nombre_tirage=10000):
    eigvals_df = pd.DataFrame(np.zeros((nombre_tirage, d)))
    for tirage in range(nombre_tirage):
        cov_estim = sample_generation(mean,cov)[2]
        eigvals_estim = np.linalg.eigvals(cov_estim)
        eigvals_df.loc[tirage] = eigvals_estim
    return eigvals_df

def markowitz_portfolio(inv_cov,mean): #Donne le couple min_variance et market
    min_variance_portfolio = inv_cov@e/(e.T@inv_cov@e)
    market_portfolio = inv_cov@mean/(e.T@inv_cov@mean)
    return min_variance_portfolio,market_portfolio

def R_sigma_computation(mean,inv_cov,multcov,risk_aversion_list): #Le paramètre multcov permet d'afficher soit théorique ou réalisée (multcov = cov), soit estimée (multcov = cov_estim)
    min_var,market=markowitz_portfolio(inv_cov,mean)
    alpha = (e.T@inv_cov@mean)/risk_aversion_list
    val_R1,val_R2 = mean@min_var,mean@market
    val_sigma1,val_sigma2,val_sigma3 = min_var.T@multcov@market,market.T@multcov@market,min_var.T@multcov@market+market.T@multcov@min_var
    R = (1-alpha)*val_R1+alpha*val_R2
    sigma = (1-alpha)*(1-alpha)*val_sigma1+alpha*alpha*val_sigma2+alpha*(1-alpha)*val_sigma3
    return R,sigma

def markowitz_front_theory(mean,cov,lambdas = 100):
    inv_cov = np.linalg.inv(cov)
    risk_aversion_list = np.linspace(1,2**8,lambdas)
    R_theory,sigma_theory = R_sigma_computation(mean,inv_cov,cov,risk_aversion_list)
    return R_theory,sigma_theory
    
def markowitz_front_realised(mean,cov,sample_size = 250,lambdas=500): #Rajouter une option pour R_estimee,sigma_estimee?
    sample,mean_estim,cov_estim= sample_generation(mean,cov,sample_size)
    inv_cov_estim = np.linalg.inv(cov_estim)
    risk_aversion_list = np.linspace(1,2**8,lambdas)
    R_realised,sigma_realised = R_sigma_computation(mean,inv_cov_estim,cov,risk_aversion_list)
    return R_realised,sigma_realised

def markowitz_monte_carlo(mean,cov,k,lambdas = 500):
    size = 2**k
    M = 10000
    R_monte_carlo,sigma_monte_carlo = np.zeros(lambdas),np.zeros(lambdas)
    a = time.time()
    for _ in range(M):
        R_realised,sigma_realised = markowitz_front_realised(mean,cov,sample_size = size,lambdas = lambdas)
        R_monte_carlo+= R_realised
        sigma_monte_carlo+= sigma_realised
    return R_monte_carlo/M,sigma_monte_carlo/M

mean,cov = mean_cov_dataframe(returns_daily)
R_theory,sigma_theory = markowitz_front_theory(mean,cov,lambdas = 500)
R_realised,sigma_realised = markowitz_front_realised(mean,cov,lambdas = 500)
#R_monte_carlo,sigma_monte_carlo = markowitz_monte_carlo(mean,cov,10, lambdas = 500)
plt.xlabel('sigma')
plt.ylabel('R')
plt.plot(sigma_theory,R_theory,color='green')
plt.plot(sigma_realised,R_realised,color='blue')
#plt.plot(sigma_monte_carlo,R_monte_carlo,color='red')
plt.plot()
plt.show()
eigvals_df = eigvals(mean,cov) #Il faudrait rajouter les valeurs propres théoriques :)
eigvals_df.hist()
plt.show()