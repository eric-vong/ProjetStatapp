import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from math import sqrt
import time

df = pd.read_csv(r'F:\Desktop\Projet_Statapp\data\CAC40.csv', sep=';',decimal=',',header=0)
start = '23-10-2010'
end = '23-10-2020'
df = df.drop(columns=['DATES','WLN FP Equity']) #On supprime les dates et la dernière colonne, entrée trop tardive
df = df.dropna() #On enlève toutes les lignes où il manque au moins une donnée
df = df.reset_index(drop=True) #On réordonne les indices, faire attention pas toujours bien si on veut calculer le return monthly
df = df.apply(lambda x: 100*x/x[0]) #On normalise 
returns_daily = df.apply(lambda x: (x/(x.shift(1))-1)) #Retours journaliers
returns_total = df.apply(lambda x: (x/x[0])-1) #Retour si on achète au début et on vend à la date t
returns_daily = returns_daily.iloc[1:] #On enlève la première ligne de NaN.
d = len(df.columns) #Nombres de colonnes

#Comparaison pour la méthode la plus rapide entre iloc[1:] et .dropna

def speed_comparison1(nombre_test = 500) :
    temps_1,temps_2 = 0,0
    for nombre_test in range(nombre_test):
        a = time.time()
        df_test = pd.read_csv(r'F:\Desktop\Projet_Statapp\data\CAC40.csv', sep=';',decimal=',',header=0)
        df_test = df_test.drop(columns=['DATES','WLN FP Equity']) #On supprime les dates et la dernière colonne, entrée trop tardive
        df_test = df_test.dropna() #On enlève toutes les lignes où il manque au moins une donnée
        df_test = df_test.reset_index(drop=True) #On réordonne les indices
        returns_daily_test = df_test.apply(lambda x: (x/(x.shift(1))-1)) #Retours journaliers
        returns_daily_test = returns_daily_test.iloc[1:]
        temps_1+= time.time() - a
    for nombre_test in range(nombre_test):
        a = time.time()
        df_test = pd.read_csv(r'F:\Desktop\Projet_Statapp\data\CAC40.csv', sep=';',decimal=',',header=0)
        df_test = df_test.drop(columns=['DATES','WLN FP Equity']) #On supprime les dates et la dernière colonne, entrée trop tardive
        df_test = df_test.dropna() #On enlève toutes les lignes où il manque au moins une donnée
        df_test = df_test.reset_index(drop=True) #On réordonne les indices
        returns_daily_test = df_test.apply(lambda x: (x/(x.shift(1))-1)) #Retours journaliers
        returns_daily_test = returns_daily_test.dropna()
        temps_2+= time.time() - a
    return (temps_1 - temps_2)/nombre_test

#speed = speed_comparison() #On a speed < 0, iloc est plus rapide que dropna, pas étonnant parce que dropna doit tout parcourir pas iloc mais on sait jamais

def speed_comparison2(nombre_test=1000):
    temps_1,temps_2 = 0,0
    for nombre_test in range(nombre_test):
        a = time.time()
        x = sqrt(252)
        temps_1+= time.time() - a
    for nombre_test in range(nombre_test):
        a = time.time()
        x = np.sqrt(252)
        temps_2+= time.time() - a
    return (temps_1 -temps_2)/nombre_test

#speed = speed_comparison2() #On a speed < 0, donc math.sqrt est plus rapide que np.sqrt

def cov_mean_dataframe(df):
    return df.cov(),df.mean()

cov,mean = cov_mean_dataframe(returns_daily)
inv_cov = np.linalg.inv(cov)
d = len(df.columns)
e = np.ones(d)

def markowitz_sol(inv_cov,mean,risk_aversion):
    min_variance_portfolio = inv_cov@e/(e.T@inv_cov@e)
    market_portfolio = inv_cov@mean/(e.T@inv_cov@mean)
    alpha = (e.T@inv_cov@mean)/risk_aversion
    return (1-alpha) * min_variance_portfolio + alpha*market_portfolio

#Tester avec min_variance_portfolio.T + (portfolio_risk.reshape(-1,1).repeat(risk_aversion.shape[0],axis=1)/risk_aversion)
#puis calculer r_theory,sigma_theory etc... peut-être plus rapide que liste en compréhension??? A demander

def sample_generation(mean,cov,sample_size = 250):
    sample = np.random.multivariate_normal(mean,cov,sample_size)
    sample_mean_estim = [np.mean(sample.T[i]) for i in range(np.shape(sample.T)[0])]
    sample_cov_estim = (sample - sample_mean_estim).T@(sample - sample_mean_estim)/(sample_size -1)
    return sample,sample_mean_estim,sample_cov_estim

#def spectrum(cov,cov_estim):
#    eigvals_theory,eigvals_estim = np.linalg.eigvals(cov),np.linalg.eigvals(cov_estim)

def markowitz_front(mean,cov,sample_size = 250,lambdas=1000):
    sample,mean_estim,cov_estim= sample_generation(mean,cov)
    inv_cov_estim = np.linalg.inv(cov_estim)
    risk_aversion_list = np.linspace(1,2**11,lambdas)
    R_theory = [mean@markowitz_sol(inv_cov,mean,risk_aversion) for risk_aversion in risk_aversion_list]
    R_efficient = [mean@markowitz_sol(inv_cov_estim,mean_estim,risk_aversion) for risk_aversion in risk_aversion_list]
    R_estim = [mean_estim@markowitz_sol(inv_cov_estim,mean_estim,risk_aversion) for risk_aversion in risk_aversion_list]
    sigma_theory = [markowitz_sol(inv_cov_estim,mean,risk_aversion).T@cov@markowitz_sol(inv_cov,mean,risk_aversion) for risk_aversion in risk_aversion_list]
    sigma_efficient = [markowitz_sol(inv_cov_estim,mean_estim,risk_aversion).T@cov@markowitz_sol(inv_cov_estim,mean_estim,risk_aversion) for risk_aversion in risk_aversion_list]
    sigma_estim = [markowitz_sol(inv_cov_estim,mean_estim,risk_aversion).T@cov_estim@markowitz_sol(inv_cov_estim,mean_estim,risk_aversion) for risk_aversion in risk_aversion_list]
    return R_theory, R_efficient, R_estim, sigma_theory,sigma_efficient,sigma_estim

R_theory,R_efficient,R_estim,sigma_theory,sigma_efficient,sigma_estim = markowitz_front(mean,cov)
plt.plot(sigma_theory,R_theory,color='green')
plt.plot(sigma_theory,R_efficient,color='blue')
#plt.plot(sigma_theory,R_estim,color='red')
plt.plot()
plt.show()