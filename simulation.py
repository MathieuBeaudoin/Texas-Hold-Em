####################################################
##################                ##################
##################   SIMULATION   ##################
##################                ##################
####################################################

from setup import *
from algo import *
import pandas as pd
import numpy as np
import math, random
from scipy.stats import norm



### INDENTIFICATION DES TYPES DE MAINS  ###
def is_flush(main):
    return max(PAQUET.iloc[main].sorte.value_counts()) >= 5

def is_straight(main):
    if len(main) < 5: return False
    # Est-ce qu'on a une suite de 5?
    try:
        valeurs, compteur = np.sort([PAQUET.iloc[i].rang for i in main]), 1
    except:
        print(main)
    if 12 in valeurs:
        valeurs = np.concatenate([[-1], valeurs])
        #print(valeurs)
    for i in range(1, len(valeurs)):
        if valeurs[i] == valeurs[i-1] + 1: compteur += 1
        elif valeurs[i] > valeurs[i-1] + 1: compteur = 1
        if compteur == 5: return True
    return False

def is_straight_flush(main):
    if len(main) < 5: return False
    if is_flush(main):
        # Est-ce que les constituantes du flush forment aussi un straight?
        sorte_dominante = PAQUET.loc[main].sorte.value_counts().index[0]
        sous_main = [i for i in main if PAQUET.iloc[i].sorte == sorte_dominante]
        return is_straight(sous_main)
    else: return False

def is_n_of_a_kind(main, n):
    return max(PAQUET.loc[main].rang.value_counts()) >= n

def is_two_pairs(main):
    return sum(PAQUET.loc[main].rang.value_counts() == 2) >= 2

def is_full_house(main):
    if is_n_of_a_kind(main, 3) == False: return False # Y-a-t-il un triple?
    else: return sum(paquet.loc[main].rang.value_counts()==2) > 0 # Si oui, y a-t-il un double?
    
# Hierarchie des types de mains
def gross_ranking(main):
    if is_straight_flush(main): return 0
    elif is_n_of_a_kind(main, 4): return 1
    elif is_full_house(main): return 2
    elif is_flush(main): return 3
    elif is_straight(main): return 4
    elif is_n_of_a_kind(main, 3): return 5
    elif is_two_pairs(main): return 6
    elif is_n_of_a_kind(main, 2): return 7
    else: return 8



### SIMULATION ###
    
def tirer_sans_remise(n, cartes_restantes=PAQUET.index):
    tirages = []
    for i in range(0,n):
        m = len(cartes_restantes)
        tirage = cartes_restantes[math.floor(m * random.random())]
        tirages.append(tirage)
        cartes_restantes = [c for c in cartes_restantes if c not in tirages]
    return tirages, cartes_restantes

def tie_breaker(mat_mains_egales):
    n = mat_mains_egales.shape[0]
    #p = 13 if carte_haute else 52
    # On retire toutes les cartes communes
    cartes_communes = np.where(mat_mains_egales.sum(0)==n)
    mat_mains_egales[:, cartes_communes] = 0
    # On extrait les valeurs des cartes restantes
    valeurs = [np.where(mat_mains_egales[i])[0] % 13 for i in range(0,n)]
    mat_valeurs = np.concatenate(valeurs).reshape(n, len(valeurs[0]))
    # On ordonne ensuite en ordre décroissant sur chaque ligne les valeurs
    mat_valeurs = np.sort(mat_valeurs, axis=1)[:, ::-1]
    # Ensuite, on élimine progressivement
    gagnants = np.array(range(0,n))
    for i in range(0, mat_valeurs.shape[1]):
        gagnants = [k for k in np.where(mat_valeurs[:,i]==max(mat_valeurs[gagnants,i]))[0]
                    if k in gagnants]
        if len(gagnants) == 1: break
    return gagnants

def victoire(table, mains, carte_haute=False):
    p = 13 if carte_haute else 52
    n = len(mains)    
    if carte_haute:
        mat_compar = MAT_COMBOS[IND_CARTE_HAUTE:, :13]
        table = [PAQUET.iloc[t].rang for t in table]
        mains = [[PAQUET.iloc[m].rang for m in main] 
                 for main in mains]
    else:
        mat_compar = MAT_COMBOS[:IND_CARTE_HAUTE, :]
    m = mat_compar.shape[0]
    # Matrice du nombre de cartes dans chaque combo
    vec_seuils = mat_compar.sum(1)
    mat_seuils = np.concatenate([vec_seuils for i in range(0,n)]).reshape(n,m)
    # Representation matricielle des cartes communes
    vec_table = np.zeros(p)
    vec_table[table] = 1
    mat_mains = np.array([vec_table for i in range(0,n)]).reshape(n,p)
    # Ajout des cartes des joueurs a leurs cartes disponibles respectives
    mat_mains[np.repeat(range(0,n),2), np.concatenate(mains)] = 1
    # Matrice de matching
    mat_matches = mat_mains.dot(mat_compar.T)
    mat_matches = mat_matches >= mat_seuils
    # Quel joueur a un match dont l'indice est le plus faible?
    matches = [np.where(mat_matches[i])[0] for i in range(0,n)]
    gagnants_potentiels = np.where([len(match) for match in matches])[0]
    if len(gagnants_potentiels) == 0:
        return victoire(table, mains, carte_haute=True)
    else:
        meilleures_combos = [np.where(mat_matches[g])[0][0] for g in gagnants_potentiels]
        rangs = COMBOS.loc[meilleures_combos].rang
        meilleurs = gagnants_potentiels[np.where(rangs == min(rangs))[0]]
        if len(meilleurs) == 1:
            return meilleurs
        else:
            return meilleurs[tie_breaker(mat_mains[meilleurs,:])]  

def un_jeu(joueur, table, n_adv, save_details=False):
    details = {'donne': np.concatenate([joueur, table])}
    jeu_cartes, mains = [c for c in PAQUET.index if c not in joueur and c not in table], [joueur]
    n_restant = 5 - len(table)
    tirages, jeu_restant = tirer_sans_remise(n_restant + 2 * n_adv, jeu_cartes)
    details['tire'] = tirages
    if n_restant > 0:
        for t in tirages[:n_restant]:
            table.append(t)
    for j in range(0, n_adv):
        debut = n_restant + 2 * j
        mains.append(tirages[debut:(debut+2)])    
    gagnant = victoire(table, mains) 
    if len(gagnant) == 1:
        if save_details: return int(gagnant[0] == 0), details
        else: return int(gagnant[0] == 0)
    else: # Split pot           
        if save_details: 
            return float(0 in gagnant) / len(gagnant), details
        else: return float(0 in gagnant) / len(gagnant)

def simulation(main, table, n_adv, ITER=100, save_details=False):
    victoires, details = 0, {}           
    for i in range(0,ITER):
        if save_details:
            jeu, details[i] = un_jeu(main, table.copy(), n_adv, save_details)
        else:
            jeu = un_jeu(main, table.copy(), n_adv, save_details)
        victoires += jeu
        if save_details: details[i]['resultat'] = jeu
    if save_details:
        return victoires/ITER, details
    else: return victoires/ITER

def genScenario(initial_states, n_adv):
    # Avec combien de cartes peut-on commencer?
    initial_state = initial_states[math.floor(len(initial_states)*random.random())]
    cartes_r = PAQUET.index
    main, cartes_r = tirer_sans_remise(2, cartes_r)
    if initial_state > 2:
        table, cartes_r = tirer_sans_remise(initial_state-2, cartes_r)
    else:
        table = []
    # Contre combien d'adversaire joue-t-on?
    if n_adv == 0:
        # On va dire que ca peut aller de 1 a 6 inclusivement
        n_adv = math.floor(random.uniform(1,7))
    return main, table, n_adv

def montecarlo(N, ITER=100, FICH_OUTPUT="simulation.csv", initial_states=[2,5,6,7], n_adv=0):    
    global MC
    COMBOS_ranked = COMBOS.loc[:,['combo', 'rang', 'type_main']]
    a = datetime.now()
    print(a)    
    MC = pd.DataFrame({'Joueur': np.repeat(None,N),
                       'Table': np.repeat(None,N),
                       'N_adv': np.repeat(None,N),
                       'Pr_sim': np.repeat(None,N),
                       'Pr_algo': np.repeat(None,N)
                      }, index = range(0,N))   
    for i in range(0,N):
        print("\n\n\n******************\nSIMULATION #"+str(i+1))
        main, table, n_adv = genScenario(initial_states, n_adv)
        # On enregistre les parametres dans notre dataframe:
        MC.at[i,'Joueur'] = main
        MC.at[i,'Table'] = table
        MC.at[i,'N_adv'] = n_adv        
        # Que nous donne la simulation?
        print_scenario(main, table, n_adv)
        MC.at[i,'Pr_sim'] = p_sim = simulation(main, table, n_adv, ITER)
        print("Simulation: "+str(p_sim*100)+"%")        
        # Que nous donnent les algorithmes?
        MC.at[i,'Pr_algo'] = p_algo = algorithme(main, table, n_adv)
        print("Algo: " + str(round(p_algo, 4)), end="\t")
        MC.at[i, 'Erreur'] = erreur = abs(p_algo - p_sim)
        print("Erreur abs: "+str(round(erreur,4)), end="\t")
        if 0 < p_algo < 1:
            z = erreur / math.sqrt(p_algo * (1-p_algo) / ITER)
            val_p = (1 - norm.cdf(z)) * 2
        elif erreur == 0:
            val_p = 1
        else:
            val_p = -1
        MC.at[i, 'ValP'] = val_p
        print("Valeur-p: " + str(round(val_p,4))) 
        print(datetime.now())     
    # Sortie des resultats       
    MC = MC.sort_values(by='ValP')
    MC.to_csv(FICH_OUTPUT, index=False, sep=";")
    print(MC.head())
    return MC