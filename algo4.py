from setup import *
import pandas as pd
import numpy as np

# Sélecteur de vecteurs
garder = lambda vec1, vec2: sum(np.in1d(vec1, vec2)) == 0

def classifRangsEgaux(rangJoueur, matIndices, joueur, vecTable):
    n = matIndices.shape[1]
    resultat = np.zeros(n)
    egalParTable = False
    combos = COMBOS.loc[COMBOS.rang==rangJoueur].combo
    carteHaute = combos.index[0] >= IND_CARTE_HAUTE
    if carteHaute:
        vecTable[np.where(vecTable)[0]%13] = 1
        vecTable[13:] = 0
        matchTable = vecTable[:13].dot(MAT_COMBOS[IND_CARTE_HAUTE:, :13].T)
        matchTable = np.where(matchTable >= LONGUEURS[IND_CARTE_HAUTE:])[0] 
    else:
        matchTable = np.where(vecTable.dot(MAT_COMBOS.T) >= LONGUEURS)[0]
    if len(matchTable) > 0:
        egalParTable = min(matchTable) in COMBOS.loc[COMBOS.rang==rangJoueur].index
    if egalParTable:
        # Il faut donc comparer la force des cartes non-partagées
        joueur = joueur % 13
        matIndicesMod = np.sort(matIndices%13, axis=0)[::-1]
        resultat[np.where(matIndicesMod[0] > max(joueur))] = 1
        egaux = np.where(matIndicesMod[0] == max(joueur))[0]
        resultat[egaux[np.where(matIndicesMod[1,egaux] > min(joueur))[0]]] = 1
        resultat[egaux[np.where(matIndicesMod[1,egaux] == min(joueur))[0]]] = 0.5
    else:
        # Il faut identifier les cartes ne servant pas à la combo de base
        p = 13 if carteHaute else 52    
        matCas = np.zeros((n+1)*p).reshape((n+1),p)
        matCas[np.repeat(np.arange(n+1),2), 
               np.concatenate([joueur, np.concatenate(matIndices.T)])%p] = 1
        matCas[:, np.where(vecTable)[0]%p] = 1
        matCombosEg = np.zeros(p*len(combos)).reshape(len(combos),p)
        matCombosEg[np.repeat(np.arange(len(combos)),len(combos.iloc[0])),
                    np.concatenate([c for c in combos])] = 1
        matCompar = matCas.dot(matCombosEg.T)   
        findMin = lambda vec: min(np.where(vec >= len(combos.iloc[0]))[0])
        matCas -= matCombosEg[np.apply_along_axis(findMin, 1, matCompar)]
        matCas[matCas < 0] = 0
        matCas[np.where(matCas)[0], np.where(matCas)[1]%13] = 1
        matCas = matCas[:,:13]
        matCasOrd = -1 * np.ones(matCas.shape[0]*8).reshape(matCas.shape[0],8)
        for i in np.arange(matCas.shape[0]):
            vec = np.sort(np.where(matCas[i])[0])[::-1]
            matCasOrd[i,:len(vec)] = vec
        #print("matCasOrd:")
        #print(matCasOrd)
        #print("matCasOrd.shape="+str(matCasOrd.shape))
        #print("joueur="+str(joueur))
        joueur, matCasOrd = matCasOrd[0], matCasOrd[1:]
        resultat[np.where(matCasOrd[:,0] > joueur[0])[0]] = 1
        egaux = np.where(matCasOrd[:,0] == joueur[0])[0]
        for i in range(1,7):
            if len(egaux) == 0: break
            """
            print("egaux (debut):"+str(egaux))
            bloc1 = np.where(matCasOrd[egaux,i] > joueur[i])[0]
            print("bloc1: "+str(bloc1))
            bloc2 = egaux[bloc1.astype('int')]
            print("bloc2: "+str(bloc2))
            bloc3 = resultat[bloc2]
            print("bloc3: "+str(bloc3))
            print("\negaux="+str(egaux))
            print("len(egaux)="+str(len(egaux)))
            print("len(resultat)="+str(len(resultat)))
            print("joueur[i]="+str(joueur[i]))
            print("matCasOrd[egaux,i]>joueur[i]:")
            print(matCasOrd[egaux,i]>joueur[i])
            print("np.where(matCasOrd[egaux,i] > joueur[i])[0]="+str(np.where(matCasOrd[egaux,i] > joueur[i])[0]))
            print("int?")
            print([isinstance(k,np.int64) for k in np.where(matCasOrd[egaux,i] > joueur[i])[0]])
            print("egaux[np.where(matCasOrd[egaux,i] > joueur[i])[0]]="+str(egaux[np.where(matCasOrd[egaux,i] > joueur[i])[0]]))
            print("resultat[egaux[np.where(matCasOrd[egaux,i] > joueur[i])[0]]]="+str(resultat[egaux[np.where(matCasOrd[egaux,i] > joueur[i])[0]]]))
            """
            resultat[egaux[np.where(matCasOrd[egaux,i] > joueur[i])[0]]] = 1
            egaux = np.array([eg for eg in np.where(matCasOrd[:,i]==joueur[i])[0] if eg in egaux])
            #print("egaux(fin): "+str(egaux))
            if joueur[i+1] == -1:
                #print("icitte")
                resultat[egaux] == 0.5
                break
    return resultat

def matPlusFort(vecJoueur, vecTable, exclus):
    matPF = np.zeros(52*52).reshape(52,52)
    # Determination des combinaisons pertinentes à évaluer
    rAOb = np.where(np.triu(np.ones(52*52).reshape(52,52), 1)) # Indices à considérer
    rAOb = np.concatenate([rAOb[0],rAOb[1]]).reshape(2,len(rAOb[0])) # Formattage en matrice
    rAOb = rAOb[:,np.apply_along_axis(garder,0,rAOb,exclus)] # Retrait des exclusions
    # Creation d'une matrice de mains d'adversaires possibles, lorsque combinées à la table
    nAOb = rAOb.shape[1]
    matCombos = np.zeros(52*nAOb).reshape(nAOb,52)
    matCombos[np.repeat(range(0,nAOb),2), np.concatenate(rAOb.T)] = 1
    matCombos[:, np.where(vecTable)[0]] = 1
    # Comparaison avec la matrice des combinaisons possibles
    matCompar = matCombos.dot(MAT_COMBOS.T)
    matCompar = matCompar >= LONGUEURS
    mains_fortes = np.where(matCompar)[0]
    # Est-ce que la main du joueur correspond à une des combos possibles?
    matchJoueur = (vecJoueur + vecTable).dot(MAT_COMBOS.T)
    matchJoueur = np.where(matchJoueur >= LONGUEURS)[0]
    if len(matchJoueur) == 0: # Le joueur a une carte haute
        # Toutes les combos qui ont eu au moins un match sont donc forcément 
        # plus fortes que la main du joueur
        matPF[rAOb[0,mains_fortes], rAOb[1,mains_fortes]] = 1
        # On doit alors classifier les mains faibles d'adversaires
        mains_faibles = np.array([i for i in range(0,nAOb) if i not in mains_fortes])
        nFaibles = len(mains_faibles)
        matCombosFaibles = np.zeros(13*nFaibles).reshape(nFaibles,13)
        matCombosFaibles[:, np.where(vecTable)[0]%13] = 1
        matCombosFaibles[np.repeat(range(0,nFaibles),2), 
                         np.concatenate(rAOb[:, mains_faibles].T)%13] = 1
        matCombosFaibles[:, np.where(vecTable)[0]%13] = 1
        matCompar = matCombosFaibles.dot(MAT_COMBOS[IND_CARTE_HAUTE:,:13].T)
        matCompar = matCompar >= LONGUEURS[IND_CARTE_HAUTE:]
        rangsAdv = np.array([min(np.where(matCompar[i])[0]) for i in range(0,nFaibles)])
        # Cette dernière opération nous donne les positions dans la matrice, mais en 
        # réalité ce sont les rangs du dataframe COMBOS qui comptent:
        rangsAdv = COMBOS.iloc[IND_CARTE_HAUTE + rangsAdv].rang
        # Il reste à obtenir le rang du joueur et à comparer        
        vecJoueurMod = np.zeros(13)
        vecJoueurMod[np.where(vecJoueur + vecTable)[0]%13] = 1
        matchJoueur = vecJoueurMod.dot(MAT_COMBOS[IND_CARTE_HAUTE:,:13].T)
        matchJoueur = matchJoueur >= LONGUEURS[IND_CARTE_HAUTE:]
        rangJoueur = COMBOS.iloc[IND_CARTE_HAUTE + min(np.where(matchJoueur)[0])].rang 
        selecteur = mains_faibles
    else: # Le joueur a une main plus forte qu'une carte haute
        nFortes = len(mains_fortes)
        rangJoueur = COMBOS.iloc[min(matchJoueur)].rang
        rangsAdv = np.array([min(np.where(matCompar[i])[0]) for i in mains_fortes])
        # Conversion des indices matriciels en rangs du dataframe COMBOS
        rangsAdv = COMBOS.iloc[rangsAdv].rang
        selecteur = mains_fortes
    # On classe les combos d'adversaires
    adv_gagne = np.where(rangsAdv < rangJoueur)[0]
    matPF[rAOb[0,selecteur[adv_gagne]], rAOb[1,selecteur[adv_gagne]]] = 1    
    # Traitement des égalités, s'il y a lieu
    adv_egal = np.where(rangsAdv == rangJoueur)[0]
    if len(adv_egal) > 0:
        vecRangsEgaux = classifRangsEgaux(rangJoueur, rAOb[:, selecteur[adv_egal]], 
                                          np.where(vecJoueur)[0], vecTable)
        matPF[rAOb[0,selecteur[adv_egal]], rAOb[1,selecteur[adv_egal]]] = vecRangsEgaux
    return matPF

def matInfluence(matrice):
    n = matrice.shape[0]
    matInfl = np.zeros(n*n).reshape(n,n)
    for i in range(0,n):
        matInfl[i,i] = matrice[i,i] - sum(matrice[i,:]) - sum(matrice[:,i])
    # Ajout des croisements; symmétrique pour simplifier l'accès ultérieur
    matInfl += np.triu(matrice,1) + np.triu(matrice,1).T
    return matInfl

def recursionMatricielle(matInfl, totalBase, nAdvRestant, posMF, posSplit, posFixe=[], nFixe=0):
    # Retrait du poids probabiliste des colonnes et rangées fixées
    resultat = totalBase + sum(matInfl[posFixe, posFixe])
    # Ajustement pour ne pas retirer en double les croisements
    croisements = np.asarray(list(combinations(posFixe,2)))
    if len(posFixe) > 0:
        resultat += sum(matInfl[croisements[:,0], croisements[:,1]])
    # Récursion, s'il reste plus d'un niveau à évaluer
    if nAdvRestant > 1:
        for i in range(0, len(posMF)):
            posFixe_mod = np.concatenate([posFixe, posMF[i]])
            posMF_mod = posMF[np.apply_along_axis(garder,1,posMF,posFixe_mod)]
            resultat += recursionMatricielle(matInfl, totalBase, nAdvRestant-1, 
                                             posMF_mod, posSplit, posFixe_mod, nFixe+2)
        for i in range(0, len(posSplit)):
            posFixe_mod = np.concatenate([posFixe, posSplit[i]])
            posSplit_mod = posSplit[np.apply_along_axis(garder,1,posSplit,posFixe_mod)]
            resultat += 0.5 * recursionMatricielle(matInfl, totalBase, nAdvRestant-1,
                                                   posMF, posSplit_mod, posFixe_mod, nFixe+2)
    else:
        for pos in range(0, len(posMF)):
            i, j = posMF[pos][0], posMF[pos][1]
            resultat += sum(matInfl[[i,i,j], [i,j,j]])
        for pos in range(0, len(posSplit)):
            i, j = posSplit[pos][0], posSplit[pos][1]
            resultat += 0.5 * sum(matInfl[[i,i,j], [i,j,j]])    
    return resultat / ((43-nFixe)*(42-nFixe)/2)

def probaPlusFortTableFixe(vecJoueur, vecTable, n_adv):
    exclus = np.where(vecJoueur + vecTable)[0]
    matPF = matPlusFort(vecJoueur, vecTable, exclus)
    probaTot = matPF.sum()
    if n_adv > 1:
        matInfl = matInfluence(matPF)
        # Identification des positions dans la matrice liées à des combos d'adversaire
        # moins fortes que celle du joueur
        posMF = np.where(np.triu(np.ones(52*52).reshape(52,52) - 2 * matPF, 1) > 0)
        posMF = np.concatenate([posMF[0], posMF[1]]).reshape(2,len(posMF[0])).T
        posMF = posMF[np.apply_along_axis(garder,1,posMF,exclus)]
        # Identification des positions dans la matrice liées à des split pots
        posSplit = np.where(matPF==0.5)
        posSplit = np.concatenate([posSplit[0], posSplit[1]]).reshape(2,len(posSplit[0])).T
        # Récursion sur les emplacements où est pas déjà certain que l'adversaire gagne
        probaTot += recursionMatricielle(matInfl, probaTot, n_adv-1, posMF, posSplit)
    return probaTot / (45*44/2)

def algorithme(main, table, n_adv):
    # Transformation des indices de cartes en vecteurs indicateurs
    vecJoueur = np.zeros(52)
    vecJoueur[main] = 1
    vecTable = np.zeros(52)
    vecTable[table] = 1
    if len(table) == 5:
        return 1 - probaPlusFortTableFixe(vecJoueur, vecTable, n_adv)
    else:
        aTirer = 5 - len(table)
        tirable = np.arange(52)
        tirable = tirable[np.isin(tirable, np.concatenate([main,table]))==False]
        possibles = np.asarray(list(combinations(tirable,aTirer)))
        n = possibles.shape[0]
        probasPlusFort = np.zeros(n)
        for i in np.arange(n):
            vecTableMod = np.zeros(52) + vecTable
            vecTableMod[possibles[i]] = 1
            probasPlusFort[i] = probaPlusFortTableFixe(vecJoueur, vecTableMod, n_adv)
        return 1 - sum(probasPlusFort) / n
  
"""
main=[31, 15]
table=[46, 39, 22, 40, 30]
n_adv=1   

pistes = np.where(matPF==0)
pistes = np.concatenate([pistes[0], pistes[1]]).reshape(2,len(pistes[0]))
valide = lambda i, j: i<j
pistes = pistes[:, np.apply_along_axis(valide, 0, pistes[0], pistes[1])]
pistes % 13
"""