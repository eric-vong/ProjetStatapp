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

def markowitz_front1(mean,cov,sample_size = 250,lambdas=100): #Calcul par définition de R et sigma
    inv_cov = np.linalg.inv(cov)
    sample,mean_estim,cov_estim= sample_generation(mean,cov)
    inv_cov_estim = np.linalg.inv(cov_estim)
    risk_aversion_list = np.linspace(1,2**8,lambdas)
    R_theory = [mean@markowitz_sol for markowitz_sol in markowitz_weight(inv_cov,mean,risk_aversion_list)]
    sigma_theory = [markowitz_sol.T@cov@markowitz_sol for markowitz_sol in markowitz_weight(inv_cov,mean,risk_aversion_list)]
    return R_theory, sigma_theory

def markowitz_front2(mean,cov,sample_size = 250,lambdas=100): #Retenir en mémoire le produit matriciel
    sample,mean_estim,cov_estim= sample_generation(mean,cov,sample_size = sample_size)
    inv_cov_estim = np.linalg.inv(cov_estim)
    risk_aversion_list = np.linspace(1,2**8,lambdas)
    R_theory,sigma_theory = R_sigma_computation(mean,cov,cov,risk_aversion_list)
    return R_theory, sigma_theory

def speed_front(test=100,lambdas = 100): #On test sur 100 itérations et 100 valeurs différents de lambda la différence entre front1 et front2
    temps1,temps2 = 0,0
    for _ in range(test):
        a = time.time()
        markowitz_front1(mean,cov)
        temps1+= time.time()-a
    for _ in range(test):
        a = time.time()
        markowitz_front2(mean,cov)
        temps2+= time.time()-a
    print(temps1/test,temps2/test)

speed_front() #

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

def speed_lw(test=100,lambdas = 100): #On test sur 100 itérations et 100 valeurs différents de lambda la différence entre lw=True ou lw=False
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

def ledoit_wolf_shrink1(cov_estim,sample): #Renvoie de la matrice de covariance shrinké par ledoit wolf
    d = np.shape(cov_estim)[0]
    m = np.trace(cov_estim)/d
    d_est = np.linalg.norm(cov_estim - m*np.identity(d))**2/d
    b_barre = 0 
    for observation in sample:
        matrice_annexe = np.array([observation])
        prod = matrice_annexe.T@matrice_annexe
        b_barre+= np.linalg.norm(prod - cov_estim)**2
    b_barre = b_barre/(d*len(sample)**2)
    b = min(b_barre,d_est)
    a = d_est - b
    return b/d_est*m*np.identity(d)+a/d_est*cov_estim

def ledoit_wolf_shrink2(cov_estim,sample): #Renvoie de la matrice de covariance shrinké par ledoit wolf scikit
    lw = LedoitWolf().fit(sample)
    return lw

def speedtest_lw(mean,cov,sample_size=250,test=5000): #speedtest entre scikit et notre implémentation
    temps1,temps2 = 0,0
    sample,mean_estim,cov_estim= sample_generation(mean,cov,sample_size)
    for _ in range(test):
        a = time.time()
        ledoit_wolf_shrink1(cov_estim,sample)
        temps1+= time.time() - a
    for _ in range(test):
        a = time.time()
        ledoit_wolf_shrink2(cov_estim,sample)
        temps2+= time.time() - a
    print(temps1/test,temps2/test)

speedtest_lw(mean,cov)

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.covariance import LedoitWolf
from math import sqrt
import time

df = pd.read_csv(r'F:\Desktop\Projet_Statapp\data\CAC40.csv', sep=';',decimal=',')
#df = pd.read_csv('C:/Users/rapha/OneDrive/Documents/ENSAE Travail/2A/StatApp/ProjetStatapp/data/CAC40.csv', sep=';',decimal=',')

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

def ledoit_wolf_shrink_theory(cov,cov_estim): #Renvoie la matrice théorique de shrinkage
    d = np.shape(cov)[0]
    mu = np.trace(cov)/d
    alpha_2 = np.linalg.norm(cov - mu*np.identity(d))**2/d
    delta_2 = np.linalg.norm(cov_estim - mu*np.identity(d))**2/d
    beta_2 = delta_2 - alpha_2
    return beta_2/delta_2*mu*np.identity(d)+alpha_2/delta_2*cov_estim

def ledoit_wolf_shrink(cov_estim,sample,corr=False): #Renvoie la matrice de covariance empirique shrinké par ledoit wolf, corr=True si on souhaite passer d'abord en corrélation avant de le faire
    if corr: #On transforme covariance en corrélation
        diag_vect = np.sqrt(np.diag(cov_estim))
        diag_mat = np.diag(diag_vect)
        diag_inv = np.linalg.inv(diag_mat)
        sample = sample/diag_vect #On met la variance à 1 pour chaque échantillon
        cov_estim = diag_inv@cov_estim@diag_inv #=np.cov(sample.T)
    d = np.shape(cov_estim)[0]
    m = np.trace(cov_estim)/d
    d_est = np.linalg.norm(cov_estim - m*np.identity(d))**2/d
    b_barre = 0 
    for observation in sample:
        matrice_annexe = np.array([observation])
        prod = matrice_annexe.T@matrice_annexe
        b_barre+= np.linalg.norm(prod - cov_estim)**2
    b_barre = b_barre/(d*len(sample)**2)
    b = min(b_barre,d_est)
    a = d_est - b
    #lw = LedoitWolf().fit(sample)
    #coeff = lw.shrinkage_*d_est #b_barre scikit learn
    shrink_cov = b/d_est*m*np.identity(d)+a/d_est*cov_estim
    if corr: #On repasse de corrélation à covariance
        shrink_cov = diag_mat@shrink_cov@diag_mat
    return shrink_cov

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

def markowitz_front_theory(mean,cov,lambdas = 100): #Calcule la frontière efficiente théorique une fois pour toute
    inv_cov = np.linalg.inv(cov)
    risk_aversion_list = np.linspace(1,2**4,lambdas)
    R_theory,sigma_theory = R_sigma_computation(mean,inv_cov,cov,risk_aversion_list)
    return R_theory,sigma_theory
    
def markowitz_front_realised(mean,cov,sample_size = 250,lambdas=100,theory=True,LedoitWolf=False): #Génère un échantillon et calcule les frontières efficientes réalisées si True sinon estimées (comparées à la théorie ou non)
    sample,mean_estim,cov_estim= sample_generation(mean,cov,sample_size)
    if LedoitWolf:
        cov_estim = ledoit_wolf_shrink(cov_estim,sample)
    inv_cov_estim = np.linalg.inv(cov_estim)
    risk_aversion_list = np.linspace(1,2**4,lambdas)
    if theory:
        multcov = cov
    else :
        multcov = cov_estim
    R,sigma = R_sigma_computation(mean,inv_cov_estim,multcov,risk_aversion_list)      
    return R,sigma

def comparison_ledoit_wolf(mean,cov,sample_size = 250, lambdas = 100):
    sample,mean_estim,cov_estim= sample_generation(mean,cov,sample_size)
    cov_estim_lw_corr = ledoit_wolf_shrink(cov_estim,sample,corr=True)
    cov_estim_lw = ledoit_wolf_shrink(cov_estim,sample)
    inv_cov_estim,inv_cov_estim_lw_corr,inv_cov_estim_lw = np.linalg.inv(cov_estim),np.linalg.inv(cov_estim_lw_corr),np.linalg.inv(cov_estim_lw)
    risk_aversion_list = np.linspace(1,2**4,lambdas)
    R,sigma = R_sigma_computation(mean,inv_cov_estim,cov,risk_aversion_list)
    R_lw_corr,sigma_lw_corr = R_sigma_computation(mean,inv_cov_estim_lw_corr,cov,risk_aversion_list)
    R_lw,sigma_lw = R_sigma_computation(mean,inv_cov_estim_lw,cov,risk_aversion_list)
    return R,sigma,R_lw_corr,sigma_lw_corr,R_lw,sigma_lw

def markowitz_monte_carlo(mean,cov,k,lambdas = 100,lw=False): #Processus de Monte-Carlo
    size = 2**k
    M = 10000
    R_monte_carlo,sigma_monte_carlo = np.zeros(lambdas),np.zeros(lambdas)
    for _ in range(M):
        R_realised,sigma_realised = markowitz_front_realised(mean,cov,sample_size = size,lambdas = lambdas,LedoitWolf = lw)
        R_monte_carlo+= R_realised
        sigma_monte_carlo+= sigma_realised
    return R_monte_carlo/M,sigma_monte_carlo/M

mean,cov = mean_cov_dataframe(returns_daily) #Moyenne et covariance théorique
R_theory,sigma_theory = markowitz_front_theory(mean,cov,lambdas = 100)
#R_realised,sigma_realised = markowitz_front_realised(mean,cov)
#R_realised_lw,sigma_realised_lw = markowitz_front_realised(mean,cov,LedoitWolf=True)
#R_estimated,sigma_estimated = markowitz_front_realised(mean,cov,theory = False)
#R_monte_carlo9,sigma_monte_carlo9 = markowitz_monte_carlo(mean,cov,9,lambdas = 100)
#R_monte_carlo_lw9,sigma_monte_carlo_lw9 = markowitz_monte_carlo(mean,cov,9,lambdas = 100,lw=True)
R_non_shrink,sigma_non_shrink,R_shrink_corr,sigma_shrink_corr,R_shrink,sigma_shrink = comparison_ledoit_wolf(mean,cov,sample_size=500,lambdas = 100)

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
plt.plot(sigma_shrink_corr,R_shrink_corr,color='orange',label='shrink + corr')
plt.legend()
plt.plot()
plt.show()

sample,mean_estim,cov_estim = sample_generation(mean,cov,sample_size=250)
bins = np.append(np.linspace(0,2.5,100),np.linspace(16,24,8*40))
plt.xlabel('$\lambda$')
plt.ylabel('Ocurrence')
plt.title('Valeurs propres de la matrice théorique et de la matrice estimée')
eigvals_theory = np.linalg.eigvalsh(cov)
eigvals_estimated = np.linalg.eigvalsh(cov_estim)
plt.hist(eigvals_theory,bins=bins,color='red',label='theory')
plt.hist(eigvals_estimated,bins=bins,color='blue',label='estimated')
plt.plot()
plt.legend()
plt.show()

def eigvals_density(mean,cov,sample_size = 250,iteration=100,nombre_bins = 100):
    bins = np.linspace(0,3,nombre_bins+1) #Les valeurs propres sont comprises entre 0 et 2.5 empiriquement, on met 3 pour être sûr
    occurrence = np.zeros(nombre_bins)
    for _ in range(iteration):
        sample,mean_estim,cov_estim = sample_generation(mean,cov,sample_size)
        corr_estim = np.cov((sample/np.sqrt(np.diag(cov_estim))).T)
        sample_eigvals = np.linalg.eigvalsh(corr_estim)
        occurrence+=plt.hist(sample_eigvals,bins=bins)[0]
    return occurrence/iteration,bins[1:]

def marcenko_pastur_density(Q,std_err,points_number = 2500):
    v = std_err**2
    eigval_max = v*(1+sqrt(1/Q))**2
    eigval_min = v*(1-sqrt(1/Q))**2
    factor = Q/(2*np.pi*v)
    points = np.linspace(eigval_min,eigval_max,points_number)
    val1 = eigval_max - points
    val2 = points - eigval_min
    val = np.sqrt(val1*val2)
    rho = factor*(val/points)
    return rho,points

q = len(df)/d

density,bins = eigvals_density(mean,cov,iteration=200,nombre_bins = 100)
marcenko_pastur,points = marcenko_pastur_density(4,0.7) #Q = 4, \sigma = 0.75
plt.clf()
plt.xlabel('$\lambda$')
plt.ylabel('densité')
plt.title('Densité des valeurs propres de la matrice de corrélation empirique')
plt.bar(bins,density,color='red',label='density',width = 0.01)
plt.plot(points,marcenko_pastur,color='blue',label='marcenko-pastur')
plt.legend()
plt.show()

def denoise_rmt_chosen(cov_estim,k=0): 
    diag = np.sqrt(np.diag(cov_estim))
    diag_mat = np.diag(diag)
    diag_mat_inv = np.diag(1/diag)
    corr_estim = diag_mat_inv@cov_estim@diag_mat_inv
    eigvals,eigvect = np.linalg.eigh(corr_estim) #valeurs propres + vecteurs propres
    threshold = np.mean(eigvals[:-(k+1)]) #On prend la moyenne des 38-k valeurs propres
    eigvals[eigvals < threshold] = threshold
    eigvals = np.sum(eigvals)/d*eigvals#On renormalise, d = Tr(corr_estim)
    retour = eigvect@np.diag(eigvals)@np.linalg.inv(eigvect) #On la dédiagonalise
    return diag_mat@retour@diag_mat

def comparison_rmt(mean,cov,sample_size = 250, lambdas = 100,k=0):
    sample,mean_estim,cov_estim= sample_generation(mean,cov,sample_size)
    cov_estim_rmt = denoise_rmt_chosen(cov_estim,k)
    cov_estim_lw = ledoit_wolf_shrink(cov_estim,sample)
    cov_estim_rmt_lw = denoise_rmt_chosen(cov_estim_lw,k)
    inv_cov_estim,inv_cov_estim_rmt,inv_cov_estim_lw = np.linalg.inv(cov_estim),np.linalg.inv(cov_estim_rmt),np.linalg.inv(cov_estim_lw)
    inv_cov_estim_rmt_lw = np.linalg.inv(cov_estim_rmt_lw)
    risk_aversion_list = np.linspace(1,2**4,lambdas)
    R,sigma = R_sigma_computation(mean,inv_cov_estim,cov,risk_aversion_list)
    R_rmt,sigma_rmt = R_sigma_computation(mean,inv_cov_estim_rmt,cov,risk_aversion_list)
    R_lw,sigma_lw = R_sigma_computation(mean,inv_cov_estim_lw,cov,risk_aversion_list)
    R_rmt_lw,sigma_rmt_lw = R_sigma_computation(mean,inv_cov_estim_rmt_lw,cov,risk_aversion_list)
    return R,sigma,R_rmt,sigma_rmt,R_lw,sigma_lw,R_rmt_lw,sigma_rmt_lw

R_empi,sigma_empi,R_rmt,sigma_rmt,R_shrink,sigma_shrink,R_rmt_lw,sigma_rmt_lw = comparison_rmt(mean,cov,sample_size=250,lambdas = 100)

plt.xlabel('$\sigma_p$')
plt.ylabel('$R_p$')
plt.title('Rendements en fonction de la variance d\'un portefeuille')
plt.plot(sigma_theory,R_theory,color='black',label='theory',linewidth=5)
plt.plot(sigma_empi,R_empi,color='blue',label='empirique')
plt.plot(sigma_shrink,R_shrink,color='red',label='shrink')
plt.plot(sigma_rmt,R_rmt,color='orange',label='rmt')
plt.plot(sigma_rmt_lw,R_rmt_lw,color='purple',label='rmt + lw')
plt.legend()
plt.plot()
plt.show()