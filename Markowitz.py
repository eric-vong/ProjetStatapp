import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
#from sklearn.covariance import LedoitWolf
from math import sqrt
import time
#import scipy.stats

df = pd.read_csv(r'F:\Desktop\Projet_Statapp\data\CAC40.csv', sep=';',decimal=',')

start = '23-10-2010'
end = '23-10-2020'
df = df.drop(columns=['DATES']) #On supprime les dates
df = df.drop(columns=['WLN FP Equity'])
#df = df.drop(columns=['ABBV UN Equity','PSX UN Equity','ADI UW Equity','KMI UN Equity','HCA UN Equity','FBHS UN Equity','MDLZ UW Equity','TXN UW Equity','HII UN Equity','XYL UN Equity','MPC UN Equity','WDC UW Equity','FANG UW Equity','NOW UN Equity','FB UW Equity','APTV UN Equity'])
df = df.dropna() #On enlève toutes les lignes où il manque au moins une donnée
df = df.reset_index(drop=True) #On réordonne les indices, faire attention pas toujours bien si on veut calculer le return monthly
df = df.apply(lambda x: 100*x/x[0]) #On normalise 
returns_daily = df.apply(lambda x: (x/(x.shift(1))-1)) #Retours journaliers
returns_daily = returns_daily.iloc[1:] #On enlève la première ligne de NaN.
returns_daily = returns_daily.iloc[:-250]
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
    sample_cov_estim = np.cov(sample.T,bias=False)
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

#Problème pour le sigma frontière efficiente..

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
    #Corr_estim n'est pas une corrélation, peut etre le faire?
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
    #lw = LedoitWolf().fit(sample)
    #coeff = lw.shrinkage_*d_est #b_barre scikit learn
    shrink_corr = b/d_est*m*np.identity(d)+a/d_est*corr_estim
    return shrink_corr

def comparison_ledoit_wolf(mean,corr,sigma_star,sigma_inf = 1,sample_size = 250, N = 100):
    sample,mean_estim,corr_estim= sample_generation(mean,corr,sample_size)
    corr_estim_lw = ledoit_wolf_shrink(corr_estim,sample)
    inv_corr_estim,inv_corr_estim_lw = np.linalg.inv(corr_estim),np.linalg.inv(corr_estim_lw)
    R,sigma = R_sigma_computation(mean,inv_corr_estim,corr,sigma_star,sigma_inf,N)
    R_lw,sigma_lw = R_sigma_computation(mean,inv_corr_estim_lw,corr,sigma_star,sigma_inf,N)
    return R,sigma,R_lw,sigma_lw

def markowitz_monte_carlo(mean,cov,sigma_star,sigma_inf = 1,k,N = 100,lw=False): #Processus de Monte-Carlo
    size = 2**k
    M = 10000
    R_monte_carlo,sigma_monte_carlo = np.zeros(N),np.zeros(N)
    for _ in range(M):
        R_realised,sigma_realised = markowitz_front_realised(mean,cov,sigma_star,sigma_inf,sample_size = size,N = N,LedoitWolf = lw)
        R_monte_carlo+= R_realised
        sigma_monte_carlo+= sigma_realised
    return R_monte_carlo/M,sigma_monte_carlo/M

def eigvals_density(mean,cov,sample_size = 250,iteration=100,nombre_bins = 100): #Trace la densité empirique des valeurs propres
    bins = np.linspace(0,30,nombre_bins+1) #Les valeurs propres sont comprises entre 0 et 2.5 empiriquement, on met 3 pour être sûr
    occurrence = np.zeros(nombre_bins)
    for _ in range(iteration):
        sample,mean_estim,cov_estim = sample_generation(mean,cov,sample_size)
        corr_estim = np.cov((sample/np.sqrt(np.diag(cov_estim))).T)
        sample_eigvals = np.linalg.eigvalsh(corr_estim)
        occurrence+=plt.hist(sample_eigvals,bins=bins,density=True)[0]
    return occurrence/iteration,(bins[1:]+bins[:-1])*0.5


def fit_Q_std_err(data): #On fit l'histogramme à une densité de Marcenko Pastur
    mean = np.mean(data)
    var = np.var(data)
    #skw = scipy.stats.skew(data)
    #kurt = scipy.stats.kurtosis(data,fisher=False)
    Q = mean**2/var #Selon Variance
    #Q2 = skw**2 #Selon Skewness
    #Q3 = kurt - 2 #Selon Kurtosis
    sigma2 = mean
    #sigma22 = sqrt(std/Q1)
    #sigma23 = sqrt(std/Q)
    #sigma24 = sqrt(std/Q3)
    return Q,sigma2

def marcenko_pastur_density(Q,sigma2,points_number = 100):
    #Q,sigma2 = fit_Q_std_err(data)
    eigval_max = sigma2*(1+sqrt(1/Q))**2
    eigval_min = sigma2*(1-sqrt(1/Q))**2
    factor = Q/(2*np.pi*sigma2)   
    points = np.linspace(eigval_min,eigval_max,points_number)
    val1 = eigval_max - points
    val2 = points - eigval_min
    val = np.sqrt(val1*val2)
    rho = factor*(val/points)
    return rho,points

def denoise_rmt_chosen(cov_estim,Q):
    threshold = (1+np.sqrt(1/Q))**2
    diag = np.sqrt(np.diag(cov_estim))
    diag_mat = np.diag(diag)
    diag_mat_inv = np.diag(1/diag)
    corr_estim = diag_mat_inv@cov_estim@diag_mat_inv
    eigvals,eigvect = np.linalg.eigh(corr_estim) #valeurs propres + vecteurs propres
    moyenne = np.mean(eigvals[eigvals < threshold]) #On prend la moyenne des 38-k valeurs propres, on enlève au moins la dernière car elle va perturber bcp la moyenne
    eigvals[eigvals < threshold] = moyenne #threshold = lambda+ de marchenko_pastur plus tard?
    retour = eigvect@np.diag(eigvals)@np.linalg.inv(eigvect) #On la dédiagonalise
    return diag_mat@retour@diag_mat

def comparison_rmt(mean,corr,sigma_star,sigma_inf = 1,sample_size = 250, N = 100): #RMT, LedoitWolf et estimateur empirique sur le même sample
    sample,mean_estim,corr_estim= sample_generation(mean,corr,sample_size)
    Q = sample_size/d
    corr_estim_rmt = denoise_rmt_chosen(corr_estim,Q)
    corr_estim_lw = ledoit_wolf_shrink(corr_estim,sample)
    corr_estim_rmt_lw = denoise_rmt_chosen(corr_estim_lw,Q)
    inv_corr_estim,inv_corr_estim_rmt,inv_corr_estim_lw = np.linalg.inv(corr_estim),np.linalg.inv(corr_estim_rmt),np.linalg.inv(corr_estim_lw)
    inv_corr_estim_rmt_lw = np.linalg.inv(corr_estim_rmt_lw)
    R,sigma = R_sigma_computation(mean,inv_corr_estim,corr,sigma_star,sigma_inf,N)
    R_rmt,sigma_rmt = R_sigma_computation(mean,inv_corr_estim_rmt,corr,sigma_star,sigma_inf,N)
    R_lw,sigma_lw = R_sigma_computation(mean,inv_corr_estim_lw,corr,sigma_star,sigma_inf,N)
    R_rmt_lw,sigma_rmt_lw = R_sigma_computation(mean,inv_corr_estim_rmt_lw,corr,sigma_star,sigma_inf,N)
    return R,sigma,R_rmt,sigma_rmt,R_lw,sigma_lw,R_rmt_lw,sigma_rmt_lw

#Initialisation des données:

mean,corr=mean_cov_dataframe(returns_daily)

###Comparaison entre Ledoit Wolf, Ledoit Wolf avec passage en corrélation et estimateur empirique

R_theory,sigma_theory = markowitz_front_theory(mean,corr,10)
R_realised,sigma_realised = markowitz_front_realised(mean,corr,10,sample_size=250)
#R_estimated,sigma_estimated = markowitz_front_realised(mean,corr,theory = False)
#R_monte_carlo9,sigma_monte_carlo9 = markowitz_monte_carlo(mean,corr,9,N = 100)
#R_monte_carlo_lw9,sigma_monte_carlo_lw9 = markowitz_monte_carlo(mean,corr,9,N = 100,lw=True)
R_non_shrink,sigma_non_shrink,R_shrink,sigma_shrink = comparison_ledoit_wolf(mean,corr,10,sample_size=100,N = 100)

plt.xlabel('$\sigma_p$')
plt.ylabel('$R_p$')
plt.title('Rendements en fonction de la variance d\'un portefeuille')
plt.plot(sigma_theory,R_theory,color='black',label='theory',linewidth=5)
#plt.plot(sigma_realised,R_realised,color='blue',label='realised')
#plt.plot(sigma_realised_lw,R_realised_lw,color='orange',label='realised_lw')
#plt.plot(sigma_monte_carlo9,R_monte_carlo9,color='red',label='monte-carlo')
#plt.plot(sigma_monte_carlo_lw9,R_monte_carlo_lw9,color='green',label='monte-carlo_lw')
#plt.plot(sigma_estimated,R_estimated,color='orange',label='estimated')
plt.plot(sigma_non_shrink,R_non_shrink,color='blue',label='non shrink')
plt.plot(sigma_shrink,R_shrink,color='red',label='shrink')
plt.legend()
plt.plot()
plt.show()

###Fitting Marcenko-Pastur à la densité empirique

#density,bins = eigvals_density(mean,cov,iteration=100,nombre_bins = 50) 
Q,sigma2 = fit_Q_std_err(density) #On fit Q, sigma avec les moments
marcenko_pastur,points,val1,val2 = marcenko_pastur_density(Q,sigma2)
plt.clf()
plt.xlabel('$\lambda$')
plt.ylabel('densité')
plt.title('Densité des valeurs propres de la matrice de corrélation empirique')
plt.bar(bins,density,color='black',label='density',width = bins[1]) 
#plt.plot(points,marcenko_pastur,color='blue',label='marcenko-pastur')
plt.legend()
plt.show()

###Comparaison entre estimateur empirique, RMT, RMT sur Ledoit Wolf

R_empi,sigma_empi,R_rmt,sigma_rmt,R_shrink,sigma_shrink,R_rmt_lw,sigma_rmt_lw = comparison_rmt(mean,corr,10,sample_size=250)
#Tracer plusieurs sample_size pour mettre en évidence le bruit, changer ratio T/M
plt.clf()
plt.xlabel('$\sigma_p$')
plt.ylabel('$R_p$')
plt.title('Rendements en fonction de la variance d\'un portefeuille')
plt.plot(sigma_theory,R_theory,color='black',label='theory',linewidth=5)
plt.plot(sigma_empi,R_empi,color='blue',label='empirique')
plt.plot(sigma_shrink,R_shrink,color='red',label='shrink')
plt.plot(sigma_rmt,R_rmt,color='green',label='rmt')
plt.plot(sigma_rmt_lw,R_rmt_lw,color='brown',label='rmt + lw')
plt.legend()
plt.plot()
plt.show()