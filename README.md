# Projet Statapp

Ce projet de statistiques appliquées a été effectué en partenariat entre l'ENSAE Paris et l'Université Panthéon-Sorbonne 1.
Nous nous sommes intéréssés à l'estimation de la frontière efficiente d'après la théorie moderne du portefeuille. Pour cela, nous avons implémenté les équations des portefeuilles optimaux et nous nous sommes concentrés sur le portefeuille de minimum-variance. Seule la matrice de covariance est à estimer dans ce cas. 
Pour cela, nous avons implémenté plusieurs estimateurs mathématiques (Shrinkage Ledoit-Wolf, Random Matrix Theory (tous les deux étudiés théoriquement) et Rotational Invariant Estimator (non étudié théoriquement)).
Nous avons ensuite simulé les résultats par méthode de Monte-Carlo et nous avons effectué un backtest sur la période 2010-2020 en mettant en évidence différents comportements d'estimateurs en fonction de la grandeur d/n où d est le nombre d'actifs et n le nombre de jour de données.

Mémoire du projet :
