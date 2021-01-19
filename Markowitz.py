import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.covariance import LedoitWolf
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
returns_daily = returns_daily.iloc[1:] #On enlève la première ligne de NaN.
d = len(df.columns) #Nombre de colonnes
T = len(df) #Nombre de lignes
e = np.ones(d)

def mean_cov_dataframe(df): #On renvoie la moyenne + la covariance de la BDD
    mean = df.mean()
    cov = df.cov()
    return mean,cov

def sample_generation(mean,cov,sample_size = 250): #Génération de l'échantillon + calcul de sa moyenne + covariance (à changer selon estimateur)
    sample = np.random.multivariate_normal(mean,cov,sample_size)
    sample_mean_estim = sample.mean(0)
    sample_cov_estim = np.cov(sample.T,bias=False)
    return sample,sample_mean_estim,sample_cov_estim


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

def markowitz_front_theory(mean,cov,lambdas = 500): #Calcule la frontière efficiente théorique une fois pour toute
    inv_cov = np.linalg.inv(cov)
    risk_aversion_list = np.linspace(1,2**4,lambdas)
    R_theory,sigma_theory = R_sigma_computation(mean,inv_cov,cov,risk_aversion_list)
    sigma_theory = sigma_theory
    return R_theory,sigma_theory
    
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

def markowitz_monte_carlo(mean,cov,k,lambdas = 100,LedoitWolf = False): #Processus de Monte-Carlo
    size = 2**k
    M = 10000
    R_monte_carlo,sigma_monte_carlo = np.zeros(lambdas),np.zeros(lambdas)
    for _ in range(M):
        R_realised,sigma_realised = markowitz_front_realised(mean,cov,sample_size = size,lambdas = lambdas, lw = LedoitWolf )
        R_monte_carlo+= R_realised
        sigma_monte_carlo+= sigma_realised
    return R_monte_carlo/M,sigma_monte_carlo/M

mean,cov = mean_cov_dataframe(returns_daily) #Moyenne et covariance théorique
R_theory,sigma_theory = markowitz_front_theory(mean,cov,lambdas = 100)
R_realised,sigma_realised = markowitz_front_realised(mean,cov,lambdas = 100)
R_realised_lw,sigma_realised_lw = markowitz_front_realised(mean,cov,lambdas = 100, lw = True)
R_estimated,sigma_estimated = markowitz_front_realised(mean,cov,lambdas = 100, theory = False)
R_monte_carlo,sigma_monte_carlo = markowitz_monte_carlo(mean,cov,10, lambdas = 100)
R_monte_carlo_lw,sigma_monte_carlo_lw = markowitz_monte_carlo(mean,cov,10,lambdas = 100, LedoitWolf = True)

plt.xlabel('$\sigma_p$')
plt.ylabel('$R_p$')
plt.title('Rendements en fonction de la variance d\'un portefeuille')
plt.plot(sigma_theory,R_theory,color='black',label='theory',linewidth=5)
plt.plot(sigma_realised,R_realised,color='blue',label='realised')
plt.plot(sigma_realised_lw,R_realised_lw,color='orange',label='realised_lw')
plt.plot(sigma_monte_carlo,R_monte_carlo,color='red',label='monte-carlo')
plt.plot(sigma_monte_carlo_lw,R_monte_carlo_lw,color='green',label='monte-carlo_lw')
plt.plot(sigma_estimated,R_estimated,color='orange',label='estimated')
plt.legend()
plt.plot()
plt.show()

plt.xlabel('$\lambda$')
plt.ylabel('ocurrence')
plt.title('Valeurs propres de la matrice théorique et de la matrice estimée')
eigvals_theory = np.linalg.eigvalsh(cov)
eigvals_estimated = np.linalg.eigvalsh(sample_generation(mean,cov)[2])
bins = np.append(np.linspace(10**-5,10**-3,100),np.linspace(6*10**-3,8*10**-3,100))
plt.hist(eigvals_theory,bins=bins,color='red',label='theory')
plt.hist(eigvals_estimated,bins=bins,color='blue',label='estimated')
plt.legend()
plt.show()

def shrinkage(alpha,cov_estim): #A voir si on prend en paramètre une liste alpha ou juste un alpha
    inv_cov_estim = np.linalg.inv(cov_estim)
    prod_1 = e@e.T
    prod_2 = e.T@inv_cov_estim@e
    prod_3 = T*(T+alpha+1)
    return ((1 + 1/(T+alpha))*cov_estim + (alpha/prod_3* prod_1 / prod_2)) #Permet d'optimiser la complexité si alpha est une liste

#Pour implémenter Ledoit Wolf par nous même?

def pi_i_j(cov_estim,i,j):
    val = 0
    ligne_i_centre = df.iloc[i]-df.iloc[i].mean()
    ligne_j_centre = df.iloc[j]-df.iloc[j].mean()
    for k in range(d):
        val+= (ligne_i_centre.iloc[k]*ligne_j_centre.iloc[k] - cov_estim[i,j])**2  #((df.iloc[i] - R_i_mean)*(df.iloc[j]-R_j_mean)) peut être?
    return val/T

def pi(cov_estim):
    val = 0
    for i in range(d):
        for j in range(d):
            val += pi_i_j(cov_estim,i,j)
    return val

def mu_i_j(cov_estim,i,j):
    val = 0
    ligne_i_centre = df.iloc[i]-df.iloc[i].mean()
    ligne_j_centre = df.iloc[j]-df.iloc[j].mean()
    for k in range(d):
        val+= (ligne_i_centre.iloc[k]**2 - cov_estim[i,j])*(ligne_i_centre.iloc[k]*ligne_j_centre.iloc[k] - cov_estim[i,j])
    return val/T

def rho(cov_estim,rho_mean):
    val = 0
    for i in range(d):
        val+= pi_i_j(cov_estim,i,i)
        somme = 0
        for j in range(d):
            if i == j :
                pass
            else :
                coeff = sqrt(cov_estim[j,j]/cov_estim[i,i])
                somme+= coeff * mu_i_j(cov_estim,i,j) + 1/coeff * mu_i_j(cov_estim,j,i)
        somme = somme*rho_mean/2
        val+= somme
    return val

def phi(cov_estim,rho_mean):
    phi = np.zeros((d,d))
    for i in range(d):
        phi[i,i] = cov_estim[i,i]
        for j in range(d):
            coeff = rho_mean * sqrt(cov_estim[i,i]*cov_estim[j,j])
            phi[i,j] = coeff
            phi[j,i] = coeff
    return phi

def gamma(phi,cov_estim):
    val = 0
    for i in range(d):
        for j in range(d):
            val+= (phi[i,j] - cov_estim[i,j])**2
    return val
