# ST7-Generali

## Description du Projet

Ce code vise à répondre à la problématique suivante : comment optimiser les revalorisations du tarif de l'assurance pour chaque client afin d'augmenter la marge réalisée tout en limitant le taux de résiliation.

## Utilisation du Code 

Pour lancer le processus d'optimisation, il suffit d'exécuter le main. Plusieurs paramètres sont modulables :
- En mettant debug = True, on réalise le processus sur un jeu de données plus faible, ce qui permet de tester les programmes.
- En mettant plot_each = True, on fait des courbes pour chaque gamma sur lequel le processus d'optimisation est réalisé, ce qui permet de comparer les algorithmes à chaque itération.
- En mettant evaluate_sensitivity = True, on réalise les analyses de sensibilités sur les paramètres du modèle désirés.

## Comment installer le Projet

pip install -r requirements.txt 

Cette commande installe tous les modules requis pour exécuter le main.py .
