#####################################################
##################                 ##################
##################   PREPARATION   ##################
##################                 ##################
#####################################################

### TYPE D'EXECUTION ###

# Doit-on creer le dataframe de combos disjointes et hierarchisees?
INIT_ALL = False   # False si on possede deja un CSV contenant cette information
FICHIER_COMBOS = "COMBOS_ranked.csv"   # Nom du CSV a creer ou charger



### SETUP GENERAL ###

# Librairies
import pandas as pd
import numpy as np
import math, random
from itertools import combinations, product
from datetime import time, datetime, timedelta

# Generation du paquet de cartes
PAQUET = pd.DataFrame({'rang': np.concatenate([range(0,13) for _ in range(0,4)]),
                       'sorte': np.repeat(range(0,4), 13)})


# Pour s'y retrouver plus facilement
SORTES = ['pics', 'trefles', 'coeurs', 'carreaux']
RANGS = np.concatenate([[str(i) for i in range(2,11)],
                        ["Valet", "Dame", "Roi", "As"]])
TYPES_MAINS = ["straight flush", "4oak", "full house", "flush",
               "straight", "3oak", "two pairs", "2oak", "high"]

def traduire(indice_carte):
    return str(RANGS[PAQUET.iloc[indice_carte].rang]) + " de " + SORTES[PAQUET.iloc[indice_carte].sorte]

def print_scenario(main, table, n_adv):
    print("\n=== Main contre " + str(n_adv) + " adversaires ===\nJoueur:\t\t" + str(main))
    for c in main:
        print("\t"+traduire(c))
    print("Table:\t\t" + str(table))    
    for c in table: print("\t"+traduire(c))

# Pour trouver les indices de cartes a partir de leur sorte et rang
get_indices = lambda k, what: [i for i in PAQUET.index if PAQUET.loc[i,what]==k]
indices_tbl = pd.DataFrame({i: None for i in SORTES},
                           index = np.array(range(0,13))
                          )
for i in indices_tbl.index:
    indices_tbl.iloc[i,:4] = get_indices(i,"rang")
    
# Pour transformer une serie de combos en matrice d'indicatrices nx52
def matrice_combos(DF, indices, p=52):
    n = len(indices)
    combos = [combo for combo in DF.loc[indices].combo]
    matrice = np.zeros(n*p).reshape(n,p)
    for i in range(0,n):
        matrice[i,combos[i]] = 1
    return matrice 

# Pour rendre lisibles des vecteurs enregistres comme chaines de caracteres (pour charger des CSV)
def str2vec(DF, additions=None):
    headers = [header for header in DF if type(DF.loc[0,header])==str]
    if additions != None:
        for a in additions: headers.append(a)
    for col in headers:
        for i in DF.index:            
            if isinstance(DF.loc[i, col], float):
                DF.at[i, col] = None
            else:         
                valeur_str = DF.loc[i, col].replace(",","").replace("[","").replace("]","").split(" ")
                DF.at[i,col] = [int(v) for v in valeur_str if v != '']
    return DF    
    


### GENERATION DES COMBINAISONS DE BASE DE CHAQUE TYPE DE MAIN ###

def base_gen(k):    
    COMBO_GEN = []    
    # Straight flush
    if k == 0: 
        for j in range(0,len(PAQUET)-4):
            # Pour les ajouter les combos commencant par un as bas
            if j%13==0: COMBO_GEN.append(list(np.concatenate([[k for k in range(j,j+4)], [j+12]])))
            # Pour les autres cas
            if PAQUET.iloc[j].sorte == PAQUET.iloc[j+4].sorte:
                COMBO_GEN.append([k for k in range(j,j+5)])    
    # N-of-a-kind
    elif k in [1,5,7]:
        indices_noak = {1:4, 5:3, 7:2}
        n = indices_noak[k]
        for i in range(0,13):
            indices = indices_tbl.iloc[i]
            combos = list(combinations(indices,n))
            for combo in combos:
                COMBO_GEN.append([c for c in combo])            
    # Full house
    elif k == 2:
        triples, doubles = [], []
        for i in range(0,13):
            triples.append(np.asarray(list(combinations(indices_tbl.iloc[i], 3))))
            doubles.append(np.asarray(list(combinations(indices_tbl.iloc[i], 2))))
        for i in range(0,13):
            for j in range(0,13):
                if i==j: continue
                for t in triples[i]:
                    for d in doubles[j]:
                        COMBO_GEN.append(np.concatenate([t, d]))    
    # Flush
    elif k == 3:
        for i in range(0,4):
            for combo in list(combinations(indices_tbl.iloc[:,i], 5)):
                if is_straight(np.asarray(combo)): continue
                else: COMBO_GEN.append(np.asarray(combo))    
    # Straight
    elif k == 4:
        # On commence par generer les series de valeurs de cartes pertinentes
        series = []
        for i in range(-1,9):
            if i == -1: series.append([j for j in np.concatenate([[12], range(0,5)])])
            else: series.append([j for j in range(i,i+5)])        
        # Pour chaque serie, on determine quels indices sont valides pour chaque position,
        # puis on genere les combinaisons possibles
        for serie in series:
            inds = []
            for rang in serie:
                inds.append(np.asarray(indices_tbl.iloc[rang]))
            for combo in list(product(inds[0], inds[1], inds[2], inds[3], inds[4])):
                if is_flush(np.asarray(combo)): continue
                COMBO_GEN.append(np.asarray(combo))    
    # Double paire
    elif k == 6:
        for i, j in list(combinations(range(0,13),2)):
            if i == j: continue
            inds = [indices_tbl.iloc[i], indices_tbl.iloc[j]] # Listes d'indices...
            combos = product(list(combinations(inds[0],2)),
                             list(combinations(inds[1],2)))
            for c in combos: COMBO_GEN.append(np.concatenate([c[0],c[1]]))    
    # Carte haute: base sur les valeurs plutot que les indices
    elif k == 8:
        for combo in list(combinations(range(0,13), 5)):
            combo = np.sort(combo)[::-1]
            if min(combo) == max(combo)-4: continue
            else: COMBO_GEN.append(combo)    
    # On trie les elements de chaque combo pour faciliter la comparaison
    for i, combo in enumerate(COMBO_GEN):
        if k != 8: COMBO_GEN[i] = np.sort(combo)    
    return COMBO_GEN

# Fonction génératrice
def init_omegas():
    COMBOS = []
    for k, type_main in enumerate(TYPES_MAINS):
        print(type_main, end=" -> ")
        COMBOS.append(base_gen(k))
        print(str(len(COMBOS[k])) + " combinaisons")
    return COMBOS



### HIERARCHISATION DES MAINS ###

# Definie par ordre inverse -> 0 est la meilleure main possible
def hierarchie_interne(combo, k): 
    # Straight flush, straight
    if k in [0, 4]:
        valeurs = np.asarray(PAQUET.iloc[combo].rang)
        if 12 in valeurs and 0 in valeurs:
            valeur_max = 3 # As bas
        else: valeur_max = max(valeurs)
        return 12 - valeur_max    
    # N-of-a-kind
    elif k in [1, 5, 7]:
        return 12 - max(PAQUET.iloc[combo].rang)            
    # Full house
    elif k == 2:
        triple = PAQUET.iloc[combo].rang.value_counts().index[0]
        double = PAQUET.iloc[combo].rang.value_counts().index[1]
        triples = [c for c in combo if PAQUET.iloc[c].rang==triple]
        doubles = [c for c in combo if PAQUET.iloc[c].rang==double]
        rang_triples = hierarchie_interne(triples, 5)
        rang_doubles = hierarchie_interne(doubles, 7)
        return rang_triples*13 + rang_doubles    
    # Flush, carte haute
    elif k in [3, 8]:
        # On a besoin des valeurs des cartes, en ordre decroissant
        if k == 8: # Carte haute, c'est simple... On a la valeur direct
            rangs = 12 - np.sort(combo)[::-1] 
        elif k == 3: # Pour un flush il faut extraire les valeurs a partir des indices de cartes
            rangs = 12 - np.sort(np.asarray([r for r in PAQUET.iloc[combo].rang]))[::-1]
        # On initialise
        rang = 0 
        for i, c in enumerate(rangs):
            rang += c * (12 - i) ** (4-i) # Il va y avoir des sauts mais c'est pas grave
        return rang    
    # Double paire
    elif k == 6:
        haute, basse = max(PAQUET.iloc[combo].rang), min(PAQUET.iloc[combo].rang)
        rang_haute = hierarchie_interne([c for c in combo if PAQUET.iloc[c].rang==haute], 7)
        rang_basse = hierarchie_interne([c for c in combo if PAQUET.iloc[c].rang==basse], 7)
        return rang_haute*13 + rang_basse
        
def hierarchie_globale():
    COMBOS_ranked = pd.DataFrame({'combo': [], 'rang': [], 'type_main': []})
    print("En train de hierarchiser les:", end="\t")
    for k in range(0, len(COMBOS)):
        print(TYPES_MAINS[k] + "...", end="   ")
        prev_max = int(max(COMBOS_ranked.rang) + 1 if k > 0 else 0)
        combos = pd.DataFrame({'combo': [c for c in COMBOS[k]],
                               'rang': [hierarchie_interne(c,k) + prev_max for c in COMBOS[k]],
                               'type_main': np.repeat(k, len(COMBOS[k]))}
                             ).sort_values(by='rang')
        COMBOS_ranked = pd.concat([COMBOS_ranked, combos])
    
    COMBOS_ranked.rang = COMBOS_ranked.rang.astype('int')
    COMBOS_ranked.type_main = COMBOS_ranked.type_main.astype('int')
    
    return COMBOS_ranked.set_index('combo').reset_index()



### INITIALISATION ###

if INIT_ALL:
    print("Generation des combinaisons de base:")
    COMBOS = init_omegas()
    print(str(sum([len(c) for c in COMBOS])) + " combinaisons de base au total")
    print("\nHierarchisation des combinaisons:")
    COMBOS = hierarchie_globale()
    print("Hierarchisation terminee.")    
    COMBOS.to_csv(FICHIER_COMBOS, sep=";", index=False)    
else:    
    COMBOS = pd.read_csv(FICHIER_COMBOS, sep=";")
    COMBOS = str2vec(COMBOS)
    COMBOS.at[:,['rang', 'type_main']] = COMBOS.loc[:,['rang', 'type_main']].astype('int')

MAT_COMBOS = matrice_combos(COMBOS, COMBOS.index)
LONGUEURS = MAT_COMBOS.sum(1).astype('int')
VEC_RANGS = np.asarray(COMBOS.rang)
IND_CARTE_HAUTE = min(COMBOS.loc[COMBOS.type_main==8].index)