# 🌀 LB-WBS : Lattice Boltzmann for Heterogeneous Porous Media

Ce projet implémente un **schéma Lattice Boltzmann (LB-WBS)** pour la simulation d’écoulements dans des milieux poreux hétérogènes, avec dépendance en pression.

## 📂 Contenu du projet

- `LB_WBS.py` : Implémentation du schéma LB-WBS.  
- `PorousMedia.py` : Génération de matrices poreuses hétérogènes.  
- `extractJSON.py` : Lecture et extraction des paramètres à partir de fichiers `.json`.  
- `utils.py` : Fonctions utilitaires (affichage, paramètres, etc.).  
- `media_json/` : Répertoire contenant des fichiers `.json` décrivant les propriétés des couches poreuses.  
- `main.py` (exemple) : Script principal illustrant l’utilisation du schéma LB-WBS.  

## ⚙️ Dépendances

Installer les bibliothèques nécessaires (testé avec Python 3.9) :  

```bash
pip install -r requirements.txt
```

## 🚀 Exemple d’exécution

```bash
python main.py
```

Ce script :  
1. Génère une matrice poreuse à plusieurs couches.  
2. Extrait les paramètres des fichiers `.json`.  
3. Construit le modèle LB-WBS avec dépendance en pression.  
4. Lance la simulation jusqu’à convergence.  
5. Affiche la magnitude du champ de vitesse.
