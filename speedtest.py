import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from math import sqrt
import time
from sklearn.covariance import LedoitWolf

df = pd.read_csv(r'F:\Desktop\Projet_Statapp\data\CAC40.csv', sep=';',decimal=',',header=0)
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

def mean_cov_dataframe(df):
    mean,cov=df.mean(),df.cov()
    return mean,cov

def sample_generation(mean,cov,sample_size = 250):
    sample = np.random.multivariate_normal(mean,cov,sample_size)
    sample_mean_estim = sample.mean(0)
    sample_cov_estim = (sample - sample_mean_estim).T@(sample - sample_mean_estim)/(sample_size -1)
    return sample,sample_mean_estim,sample_cov_estim

mean,cov = mean_cov_dataframe(returns_daily)

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
        _ = sqrt(252)
        temps_1+= time.time() - a
    for nombre_test in range(nombre_test):
        a = time.time()
        _ = np.sqrt(252)
        temps_2+= time.time() - a
    return (temps_1 -temps_2)/nombre_test

#speed = speed_comparison2() #On a speed < 0, donc math.sqrt est plus rapide que np.sqrt

def markowitz_portfolio(inv_cov,mean): #Donne le couple min_variance et market
    min_variance_portfolio = inv_cov@e/(e.T@inv_cov@e)
    market_portfolio = inv_cov@mean/(e.T@inv_cov@mean)
    return min_variance_portfolio,market_portfolio

def markowitz_weight(inv_cov,mean,risk_aversion_list):
    min_variance_portfolio,market_portfolio = markowitz_portfolio(inv_cov,mean)
    alpha = (e.T@inv_cov@mean)/risk_aversion_list
    return ((1-alpha) * min_variance_portfolio.reshape(-1,1).repeat(risk_aversion_list.shape[0],axis=1) + alpha*market_portfolio.reshape(-1,1).repeat(risk_aversion_list.shape[0],axis=1)).T #Permet de tout vectorialiser, pas besoin de recalculer min_variance à chaque fois

def R_sigma_computation(mean,inv_cov,multcov,risk_aversion_list): #Le paramètre multcov permet d'afficher soit théorique ou réalisée (multcov = cov), soit estimée (multcov = cov_estim)
    min_var,market=markowitz_portfolio(inv_cov,mean)
    alpha = (e.T@inv_cov@mean)/risk_aversion_list
    val_R1,val_R2 = mean@min_var,mean@market
    val_sigma1,val_sigma2,val_sigma3 = min_var.T@multcov@market,market.T@multcov@market,min_var.T@multcov@market+market.T@multcov@min_var
    R = (1-alpha)*val_R1+alpha*val_R2
    sigma = (1-alpha)*(1-alpha)*val_sigma1+alpha*alpha*val_sigma2+alpha*(1-alpha)*val_sigma3
    return R,sigma 

def markowitz_front1(mean,cov,sample_size = 250,lambdas=100):
    inv_cov = np.linalg.inv(cov)
    sample,mean_estim,cov_estim= sample_generation(mean,cov)
    inv_cov_estim = np.linalg.inv(cov_estim)
    risk_aversion_list = np.linspace(1,2**8,lambdas)
    R_theory = [mean@markowitz_sol for markowitz_sol in markowitz_weight(inv_cov,mean,risk_aversion_list)]
    sigma_theory = [markowitz_sol.T@cov@markowitz_sol for markowitz_sol in markowitz_weight(inv_cov,mean,risk_aversion_list)]
    return R_theory, sigma_theory

def markowitz_front2(mean,cov,sample_size = 250,lambdas=100):
    sample,mean_estim,cov_estim= sample_generation(mean,cov,sample_size = sample_size)
    inv_cov_estim = np.linalg.inv(cov_estim)
    risk_aversion_list = np.linspace(1,2**8,lambdas)
    R_theory,sigma_theory = R_sigma_computation(mean,cov,risk_aversion_list)
    return R_theory, sigma_theory

def markowitz_front_realised(mean,cov,sample_size = 250,lambdas=100,theory=True,lw=False): #Génère un échantillon et calcule les frontières efficientes réalisées si True sinon estimées (comparées à la théorie ou non)
    sample,mean_estim,cov_estim= sample_generation(mean,cov,sample_size)
    if lw : #Si on estime via Ledoit Wolf, éventuellement si plus à ajouter on remplace cov_estim dans le calcul en le mettant en argument dans la fonction
        LW = LedoitWolf().fit(sample)
        cov_estim = LW.covariance_
    inv_cov_estim = np.linalg.inv(cov_estim)
    risk_aversion_list = np.linspace(1,2**4,lambdas)
    if theory:
        multcov = cov
    else :
        multcov = cov_estim
    R,sigma = R_sigma_computation(mean,inv_cov_estim,multcov,risk_aversion_list)      
    return R,sigma

mean,cov = mean_cov_dataframe(df)

def speed(test=100,lambdas = 100):
    temps1,temps2 = 0,0
    for _ in range(test):
        a = time.time()
        markowitz_front_realised(mean,cov,lw=True)
        temps1+= time.time()-a
    for _ in range(test):
        a = time.time()
        markowitz_front_realised(mean,cov)
        temps2+= time.time()-a
    print(temps1/test,temps2/test)

speed()

def test_affectation(test=10000,condition = False):
    temps1,temps2 = 0,0
    var1,var2 = cov, cov+1
    for _ in range(test):
        a = time.time()
        if condition:
            x = var1
        else :
            x = var2
        temps1+= time.time() - a
    for _ in range(test):
        a = time.time()
        x = condition * var1 + (1-condition)*var2
        temps2+= time.time() - a
    print(temps1/test,temps2/test)

#test_affectation() #Résultat pas étonnant, le deuxième test doit évaluer deux fois condition