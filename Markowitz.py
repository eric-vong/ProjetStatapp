import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

df = pd.read_csv(r'F:\Desktop\Projet_Statapp\data\CAC40.csv', sep=';',decimal=',',header=0)
start = '23-10-2010'
end = '23-10-2020'
df = df.drop(columns=['DATES','WLN FP Equity']) #On supprime les dates et la dernière colonne, entrée trop tardive
df = df.dropna() #On enlève toutes les lignes où il manque au moins une donnée
df = df.drop([0]) #On finit de nettoyer
cols = df.columns[df.dtypes.eq(object)]
df = df.apply(lambda x: x.str.replace(',','.')) #Pandas est censé remplacer les , en . mais il ne le fait pas à cause de la première ligne, que faire???
df[cols] = df[cols].astype(float) #On convertit le type
returns = df.apply(lambda x: (x/(x.shift(1))-1)) #Retours journaliers
returns = returns.iloc[1:]
df = df.div(df.iloc[0]/100) #On normalise pour passer aux rendements à partir des prix

cov_matrix = df.cov()
mean_vector = df.mean()