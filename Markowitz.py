import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.covariance import LedoitWolf
from math import sqrt
import time
from cmath import sqrt as csqrt
#import scipy.stats

df = pd.read_csv(r'F:\Desktop\Projet_Statapp\data\SPX_Data.csv', sep=';',decimal=',')


start = '23-10-2010'
end = '23-10-2020'
df = df.drop(columns=['DATES']) #On supprime les dates
#df = df.drop(columns=['WLN FP Equity'])
df = df.drop(columns=['ABBV UN Equity','PSX UN Equity','ADI UW Equity','KMI UN Equity','HCA UN Equity','FBHS UN Equity','MDLZ UW Equity','TXN UW Equity','HII UN Equity','XYL UN Equity','MPC UN Equity','WDC UW Equity','FANG UW Equity','NOW UN Equity','FB UW Equity','APTV UN Equity'])
df = df.dropna() #On enlève toutes les lignes où il manque au moins une donnée
df = df.reset_index(drop=True) #On réordonne les indices, faire attention pas toujours bien si on veut calculer le return monthly
df = df.apply(lambda x: 100*x/x[0]) #On normalise 
returns_daily = df.apply(lambda x: (x/(x.shift(1))-1)) #Retours journaliers
returns_daily = returns_daily.iloc[1:] #On enlève la première ligne de NaN.
d = len(df.columns) #Nombre de colonnes
T = len(df) #Nombre de lignes
e = np.ones(d)
x = np.shape(returns_daily)[0]
std = returns_daily.std()
returns_daily = returns_daily/std #On met l'écart type à 1 (diag de la cov = 1)

def mean_cov_dataframe(df): #On renvoie la moyenne + la covariance de la BDD
    mean = df.mean()
    corr = df.corr()
    return mean,corr

def sample_generation(mean,corr,sample_size = 250): #Génération de l'échantillon + calcul de sa moyenne + covariance (à changer selon estimateur)
    sample = np.random.multivariate_normal(mean,corr,sample_size)
    sample_mean_estim = sample.mean(0)
    sample_cov_estim = np.cov(sample.T,bias=True)
    return sample,sample_mean_estim,sample_cov_estim

def markowitz_portfolio(inv_corr,mean): #Donne le couple min_variance et market
    min_variance_portfolio = inv_corr@e/(e.T@inv_corr@e)
    market_portfolio = inv_corr@mean/(e.T@inv_corr@mean)
    return min_variance_portfolio,market_portfolio

def lambda_compute(mean,inv_corr,multcorr,sigma_star):
    val1,val2 = inv_corr@e*1/(e.T@inv_corr@e),inv_corr@(mean-e*(e.T@inv_corr@mean)/(e.T@inv_corr@e))
    a = val2.T@multcorr@val2
    b = 2*val2.T@multcorr@val1
    c = val1.T@multcorr@val1 - sigma_star
    delta = b**2 - 4*a*c
    x1,x2 = (-b+sqrt(delta))/(2*a),(-b-sqrt(delta))/(2*a)
    res = a*x1**2+b*x1+c
    lambda_inf = 1/x1
    return lambda_inf

def R_sigma_computation(mean,inv_corr,multcorr,sigma_star,sigma_inf, N): #Le paramètre multcov permet d'afficher soit théorique ou réalisée (multcov = cov), soit estimée (multcov = cov_estim)
    min_var,market=markowitz_portfolio(inv_corr,mean)
    lambda_inf = lambda_compute(mean,inv_corr,multcorr,sigma_star)
    lambda_sup = lambda_compute(mean,inv_corr,multcorr,sigma_inf)
    risk_aversion_list = np.linspace(lambda_inf,lambda_sup,N)
    alpha = (e.T@inv_corr@mean)/risk_aversion_list
    val_R1,val_R2 = min_var.T@mean,market.T@mean
    val_sigma1 = min_var.T@multcorr@min_var
    val_sigma2 = market.T@multcorr@market
    val_sigma3 = min_var.T@multcorr@market+market.T@multcorr@min_var
    R = (1-alpha)*val_R1+alpha*val_R2
    sigma = (1-alpha)*(1-alpha)*val_sigma1+alpha*alpha*val_sigma2+alpha*(1-alpha)*val_sigma3
    return R,sigma 

def markowitz_front_theory(mean,corr,sigma_star,sigma_inf=1, N=100): #Calcule la frontière efficiente théorique une fois pour toute
    inv_corr = np.linalg.inv(corr)
    R_theory,sigma_theory = R_sigma_computation(mean,inv_corr,corr,sigma_star,sigma_inf,N)
    return R_theory,sigma_theory

def markowitz_front_realised(mean,corr,sigma_star,sigma_inf=1,sample_size = 250,N=100,theory=True,LedoitWolf=False): #Génère un échantillon et calcule les frontières efficientes réalisées si True sinon estimées (comparées à la théorie ou non)
    sample,mean_estim,corr_estim= sample_generation(mean,corr,sample_size)
    if LedoitWolf:
        corr_estim = ledoit_wolf_shrink(corr_estim,sample)
    inv_corr_estim = np.linalg.inv(corr_estim)
    if theory:
        multcorr = corr
    else :
        multcorr = corr_estim
    R,sigma = R_sigma_computation(mean,inv_corr_estim,multcorr,sigma_star,sigma_inf, N)
    return R,sigma

def ledoit_wolf_shrink(corr_estim,sample): #Renvoie la matrice de covariance empirique shrinké par ledoit wolf, corr=True si on souhaite passer d'abord en corrélation avant de le faire
    d = np.shape(corr_estim)[0]
    m = np.trace(corr_estim)/d
    d_est = np.linalg.norm(corr_estim - m*np.identity(d))**2/d
    b_barre = 0 
    for observation in sample:
        matrice_annexe = np.array([observation])
        prod = matrice_annexe.T@matrice_annexe
        b_barre+= np.linalg.norm(prod - corr_estim)**2
    b_barre = b_barre/(d*len(sample)**2)
    b = min(b_barre,d_est)
    a = d_est - b
    lw = LedoitWolf().fit(sample)
    #coeff = lw.shrinkage_#*d_est #b_barre scikit learn
    #cov_lw = lw.covariance_
    shrink_corr = b/d_est*m*np.identity(d)+a/d_est*corr_estim
    #print(shrink_corr,cov_lw)
    #print(lw.shrinkage_,b/d_est)
    return shrink_corr

def denoise_rmt_chosen(cov_estim,Q,with_inf = False):
    threshold_sup = (1+np.sqrt(1/Q))**2
    threshold_inf = 0.
    if with_inf :
        threshold_inf = (1-np.sqrt(1/Q))**2
    diag = np.sqrt(np.diag(cov_estim))
    diag_mat = np.diag(diag)
    diag_mat_inv = np.diag(1/diag)
    corr_estim = diag_mat_inv@cov_estim@diag_mat_inv
    eigvals,eigvect = np.linalg.eigh(corr_estim) #valeurs propres + vecteurs propres
    moyenne = np.mean(eigvals[(eigvals < threshold_sup) & (eigvals > threshold_inf)]) #On prend la moyenne des valeurs propres
    eigvals[(eigvals < threshold_sup) & (eigvals > threshold_inf)] = moyenne #Pour conserver la trace
    retour = eigvect@np.diag(eigvals)@np.linalg.inv(eigvect) #On la dédiagonalise
    return diag_mat@retour@diag_mat

def stieltjes(z,q,lambda_sup,sigma2):
    lambda_plus = lambda_sup*((1+csqrt(q))/(1-csqrt(q)))**2
    lambda_plus = complex(lambda_plus,0)
    nominator = z -csqrt((z-lambda_sup)*(z-lambda_plus))
    return (nominator+ sigma2*(q-1))/(2*sigma2)

stieltjes_marcenko_pastur = np.vectorize(stieltjes)
    
def RIE(corr,sample):
    N = len(corr)
    q = N/len(sample)
    eigvals,eigvect = np.linalg.eigh(corr)
    lambda_sup = eigvals[-1]
    z = np.zeros(N,dtype=complex)
    s = np.zeros(N,dtype=complex)
    for k in range(N) :
        z[k] = complex(eigvals[k],-1/sqrt(N))
        s[k] = np.trace(z[k]*np.identity(N)-corr)
    s = 1/s
    s*= 1/N
    sigma2 = lambda_sup/(1-sqrt(q))**2
    khi_RIE = eigvals/abs(1-q+q*z*s)**2
    gamma = abs(1-q+stieltjes_marcenko_pastur(z,q,lambda_sup,sigma2))**2/eigvals * sigma2
    for val in gamma:
        val = max(val,1)
    khi_RIE = gamma*khi_RIE
    retour = eigvect@np.diag(khi_RIE)@np.linalg.inv(eigvect)
    return retour

def gmp(z,q,sigma2,lambdaN):
    lamb=lambdaN*((1+csqrt(q))/(1-csqrt(q)))
    g=(z+sigma2*(q-1)-csqrt(z-lambdaN)*csqrt(z-lamb))/(2*q*z*sigma2)
    return g

def rieCov(returns):
    cov=np.cov(returns.T)
    diag = np.sqrt(np.diag(cov))
    diag_mat = np.diag(diag)
    diag_mat_inv = np.diag(1/diag)
    estCorr = diag_mat_inv@cov@diag_mat_inv
    eigVals, eigVects=np.linalg.eig(estCorr)
    eigVals2=eigVals
    N=len(cov)
    q=N/len(returns)
    lambdaN=np.max(eigVals)
    sigma2=lambdaN/(1-sqrt(q))
    for k in range(N):
        zk=eigVals[k]-1j/sqrt(N)
        sk=0
        for i in range(N):
            if i!=k:
                sk=sk+1/(zk+eigVals[i])
        sk=sk/N
        Ek=eigVals[k]/(abs(1-q+q*zk*sk)**2)
        gammak=sigma2*abs(1-q+q*zk*gmp(zk,q,sigma2,lambdaN))
        if gammak>1:
            eigVals2[k]=gammak*Ek
        else:
            eigVals2[k]=Ek
    estCorr=eigVects@np.diag(eigVals2)@np.linalg.inv(eigVects)
    return diag_mat@estCorr@diag_mat

def markowitz_monte_carlo(mean,cov,sigma_star,k,sigma_inf = 1,N = 100,lw=False): #Processus de Monte-Carlo
    size = 2**k
    M = 10000
    R_monte_carlo,sigma_monte_carlo = np.zeros(N),np.zeros(N)
    for _ in range(M):
        R_realised,sigma_realised = markowitz_front_realised(mean,cov,sigma_star,sigma_inf,sample_size = size,N = N,LedoitWolf = lw)
        R_monte_carlo+= R_realised
        sigma_monte_carlo+= sigma_realised
    return R_monte_carlo/M,sigma_monte_carlo/M

def comparison(mean,corr,sigma_star,sigma_inf = 1,sample_size = 250, N = 100,with_inf = False): #RMT, LedoitWolf et estimateur empirique sur le même sample
    sample,mean_estim,corr_estim= sample_generation(mean,corr,sample_size)
    Q = sample_size/d
    corr_estim_rmt = denoise_rmt_chosen(corr_estim,Q,with_inf)
    corr_estim_rmt2 = denoise_rmt_chosen(corr_estim,Q,with_inf = True)
    corr_estim_lw = ledoit_wolf_shrink(corr_estim,sample)
    corr_estim_RIE = rieCov(sample)
    inv_corr_estim = np.linalg.inv(corr_estim)
    inv_corr_estim_rmt = np.linalg.inv(corr_estim_rmt)
    inv_corr_estim_rmt2 = np.linalg.inv(corr_estim_rmt2)
    inv_corr_estim_lw = np.linalg.inv(corr_estim_lw)
    inv_corr_estim_RIE = np.linalg.inv(corr_estim_RIE)
    R,sigma = R_sigma_computation(mean,inv_corr_estim,corr,sigma_star,sigma_inf,N)
    R_rmt,sigma_rmt = R_sigma_computation(mean,inv_corr_estim_rmt,corr,sigma_star,sigma_inf,N)
    R_rmt2,sigma_rmt2 = R_sigma_computation(mean,inv_corr_estim_rmt2,corr,sigma_star,sigma_inf,N)
    R_lw,sigma_lw = R_sigma_computation(mean,inv_corr_estim_lw,corr,sigma_star,sigma_inf,N)
    R_RIE,sigma_RIE = R_sigma_computation(mean,inv_corr_estim_RIE,corr,sigma_star,sigma_inf,N)
    return R,sigma,R_rmt,sigma_rmt,R_lw,sigma_lw,R_RIE,sigma_RIE,R_rmt2,sigma_rmt2

#Initialisation des données:

mean,corr=mean_cov_dataframe(returns_daily)

###Comparaison entre Ledoit Wolf, Ledoit Wolf avec passage en corrélation et estimateur empirique

R_theory,sigma_theory = markowitz_front_theory(mean,corr,10)
R_realised,sigma_realised = markowitz_front_realised(mean,corr,10,sample_size=1000)
#R_estimated,sigma_estimated = markowitz_front_realised(mean,corr,theory = False)
R_monte_carlo7,sigma_monte_carlo7 = markowitz_monte_carlo(mean,corr,10,7,N = 100)


plt.xlabel('$\sigma_p$')
plt.ylabel('$R_p$')
plt.title('Rendements en fonction de la variance d\'un portefeuille')
plt.plot(sigma_theory,R_theory,color='black',label='theory',linewidth=5)
plt.plot(sigma_realised,R_realised,color='blue',label='realised')
#plt.plot(sigma_realised_lw,R_realised_lw,color='orange',label='realised_lw')
plt.plot(sigma_monte_carlo7,R_monte_carlo7,color='red',label='monte-carlo, k = 7')
#plt.plot(sigma_estimated,R_estimated,color='orange',label='estimated')
plt.legend()
plt.plot()
plt.show()

###Comparaison entre estimateur empirique, RMT, RMT sur Ledoit Wolf

R_empi,sigma_empi,R_rmt,sigma_rmt,R_shrink,sigma_shrink,R_RIE,sigma_RIE,R_rmt2,sigma_rmt2 = comparison(mean,corr,10,sample_size=1250)
#Tracer plusieurs sample_size pour mettre en évidence le bruit, changer ratio T/M
plt.clf()
plt.xlabel('$\sigma_p$')
plt.ylabel('$R_p$')
plt.title('Rendements en fonction de la variance d\'un portefeuille')
plt.plot(sigma_theory,R_theory,color='black',label='theory',linewidth=5)
#plt.plot(sigma_empi,R_empi,color='blue',label='empirique') #Inversibilité limitée
#plt.plot(sigma_shrink,R_shrink,color='red',label='shrink')
plt.plot(sigma_rmt,R_rmt,color='green',label='rmt $\lambda_{+}$')
plt.plot(sigma_rmt2,R_rmt2,color='red',label='rmt $\lambda_{+},\lambda_{-}$')
#plt.plot(sigma_RIE,R_RIE,color='purple',label='RIE')
plt.legend()
plt.plot()
plt.show()