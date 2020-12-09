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
returns_total = df.apply(lambda x: (x-1)) #Retour si on achète au début et on vend à la date t
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
    return R_theory,sigma_theory
    
def markowitz_front_realised(mean,cov,sample_size = 250,lambdas=100,theory=True): #Génère un échantillon et calcule les frontières efficientes réalisées si True sinon estimées (comparées à la théorie ou non)
    sample,mean_estim,cov_estim= sample_generation(mean,cov,sample_size)
    inv_cov_estim = np.linalg.inv(cov_estim)
    risk_aversion_list = np.linspace(1,2**4,lambdas)
    if theory:
        multcov = cov
    else :
        multcov = cov_estim
    R,sigma = R_sigma_computation(mean,inv_cov_estim,multcov,risk_aversion_list)      
    return R,sigma

def markowitz_monte_carlo(mean,cov,k,lambdas = 100): #Processus de Monte-Carlo
    size = 2**k
    M = 10000
    R_monte_carlo,sigma_monte_carlo = np.zeros(lambdas),np.zeros(lambdas)
    for _ in range(M):
        R_realised,sigma_realised = markowitz_front_realised(mean,cov,sample_size = size,lambdas = lambdas)
        R_monte_carlo+= R_realised
        sigma_monte_carlo+= sigma_realised
    return R_monte_carlo/M,sigma_monte_carlo/M

mean,cov = mean_cov_dataframe(returns_daily)
R_theory,sigma_theory = markowitz_front_theory(mean,cov,lambdas = 100)
R_realised,sigma_realised = markowitz_front_realised(mean,cov,lambdas = 100)
R_estimated,sigma_estimated = markowitz_front_realised(mean,cov,lambdas = 100, theory = False)
#R_monte_carlo,sigma_monte_carlo = markowitz_monte_carlo(mean,cov,10, lambdas = 100)

plt.xlabel('$\sigma_p$')
plt.ylabel('$R_p$')
plt.title('Rendements en fonction de la variance d\'un portefeuille')

R_realised,sigma_realised = markowitz_front_realised(mean,cov,lambdas = 100)
plt.plot(sigma_theory,R_theory,color='green',label='oui')
plt.plot(sigma_realised,R_realised,color='blue',label='tes')
plt.plot(sigma_monte_carlo,R_monte_carlo,color='red')
#plt.plot(sigma_estimated,R_estimated,color='orange')
plt.plot()
plt.show()

def eigvals(mean,cov,nombre_tirage=100,sample_size = 250):
    eigvals_df = pd.DataFrame(np.zeros((nombre_tirage, d)))
    for tirage in range(nombre_tirage):
        cov_estim = sample_generation(mean,cov,sample_size = sample_size)[2]
        eigvals_estim = np.linalg.eigvals(cov_estim)
        eigvals_df.loc[tirage] = eigvals_estim
    return eigvals_df

eigvals_theory = np.linalg.eigvals(cov)
eigvals_estimated = np.linalg.eigvals(cov_estim)
eigvals_df = eigvals(mean,cov)
for index in range(3):
    eigvals_df.hist(column=index)
    plt.axvline(x=eigvals_theory[index],color='red')
plt.plot()
plt.show()

for index in range(len(eigvals_df.columns)):
    plt.axvline(eigvals_theory[index],color='red')
    plt.axvline(eigvals_estimated[index],color='blue')
plt.show()

vp_est=eigvals_estimated
vp=eigvals_theory
L_est=[0]*100
L_th =[0]*100
for i in range(100):
    for index in range(len(vp_est)):
        k = vp_est[index]
        j = vp[index]
        if i*10**-4<k and k<(i+1)*10**-4:
            L_est[i]+=1
        if i*10**-4<j and j<(i+1)*10**-4:
            L_th[i]+=1
M=np.linspace(0,10**-2,10**2)

plt.hist(L_est,bins=M,color='blue')
plt.hist(L_th,bins = M,color='red')
plt.ylim((0,5))
plt.show()