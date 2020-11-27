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
returns_daily = df.apply(lambda x: (x/(x.shift(1))-1)) #Retours journaliers
returns_total = df.apply(lambda x: (x[-1] / x[0]) - 1)
returns_daily = returns.iloc[1:] #On enlève la première ligne de NaN.

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

df = df.apply(lambda x: 100*x/x[0]) #On normalise pour passer aux rendements à partir des prix
cov_matrix = df.cov()
mean_vector = df.mean()

