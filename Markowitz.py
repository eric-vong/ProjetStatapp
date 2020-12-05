import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from math import sqrt
import time

df = pd.read_csv(r'F:\Desktop\Projet_Statapp\data\CAC40ORG.csv', sep=';',decimal=',')
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

def cov_mean_dataframe(df):
    return df.cov(),df.mean()

def markowitz_portfolio(inv_cov,mean): #Donne le couple min_variance et market
    min_variance_portfolio = inv_cov@e/(e.T@inv_cov@e)
    market_portfolio = inv_cov@mean/(e.T@inv_cov@mean)
    return min_variance_portfolio,market_portfolio

def markowitz_weight(inv_cov,mean,risk_aversion_list):
    min_variance_portfolio,market_portfolio = markowitz_portfolio(inv_cov,mean)
    alpha = (e.T@inv_cov@mean)/risk_aversion_list
    return ((1-alpha) * min_variance_portfolio.reshape(-1,1).repeat(risk_aversion_list.shape[0],axis=1) + alpha*market_portfolio.reshape(-1,1).repeat(risk_aversion_list.shape[0],axis=1)).T #Permet de tout vectorialiser, pas besoin de recalculer min_variance à chaque fois

#Tester avec min_variance_portfolio.T + (portfolio_risk.reshape(-1,1).repeat(risk_aversion.shape[0],axis=1)/risk_aversion)
#puis calculer r_theory,sigma_theory etc... peut-être plus rapide que liste en compréhension??? A demander

def sample_generation(mean,cov,sample_size = 250):
    sample = np.random.multivariate_normal(mean,cov,sample_size)
    sample_mean_estim = sample.mean(0)
    sample_cov_estim = (sample - sample_mean_estim).T@(sample - sample_mean_estim)/(sample_size -1)
    return sample,sample_mean_estim,sample_cov_estim

#def spectrum(cov,cov_estim):
#    eigvals_theory,eigvals_estim = np.linalg.eigvals(cov),np.linalg.eigvals(cov_estim)

def R_sigma_computation(inv_cov,mean,risk_aversion_list):
    min_var,market=markowitz_portfolio(inv_cov,mean)
    alpha = (e.T@inv_cov@mean)/risk_aversion_list
    val_R1,val_R2 = mean@min_var,mean@market
    val_sigma1,val_sigma2,val_sigma3 = min_var.T@inv_cov@market,market.T@inv_cov@market,min_var.T@inv_cov@market+market.T@inv_cov@min_var
    R = (1-alpha)*val_R1+alpha*val_R2
    sigma = (1-alpha)*(1-alpha)*val_sigma1+alpha*alpha*val_sigma2+alpha*(1-alpha)*val_sigma3
    return R,sigma

def markowitz_front1(mean,cov,sample_size = 250,lambdas=100):
    sample,mean_estim,cov_estim= sample_generation(mean,cov)
    inv_cov_estim = np.linalg.inv(cov_estim)
    risk_aversion_list = np.linspace(1,2**8,lambdas)
    R_theory = [mean@markowitz_sol for markowitz_sol in markowitz_weight(inv_cov,mean,risk_aversion_list)]
    R_realised = [mean@markowitz_sol for markowitz_sol in markowitz_weight(inv_cov_estim,mean,risk_aversion_list)]
    #R_estim = [mean_estim@markowitz_sol for markowitz_sol in markowitz_weight(inv_cov_estim,mean_estim,risk_aversion_list)]
    sigma_theory = [markowitz_sol.T@cov@markowitz_sol for markowitz_sol in markowitz_weight(inv_cov,mean,risk_aversion_list)]
    sigma_realised = [markowitz_sol.T@cov@markowitz_sol for markowitz_sol in markowitz_weight(inv_cov_estim,mean,risk_aversion_list)]
    #sigma_estim = [markowitz_sol.T@cov_estim@markowitz_sol for markowitz_sol in markowitz_weight(inv_cov_estim,mean_estim,risk_aversion_list)]
    return R_theory,R_realised, R_estim,sigma_theory,sigma_realised,sigma_estim

cov,mean = cov_mean_dataframe(returns_daily)
inv_cov = np.linalg.inv(cov)
R_theory,R_realised,R_estim,sigma_theory,sigma_realised,sigma_estim = markowitz_front(mean,cov,100)
plt.plot(sigma_theory,R_theory,color='green')
plt.plot(sigma_theory,R_realised,color='blue')
#plt.plot(sigma_theory,R_estim,color='red')
plt.plot()
plt.show()