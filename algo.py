####################################################
##################                ##################
##################   ALGORITHME   ##################
##################                ##################
####################################################

from setup import *
import pandas as pd
import numpy as np

# Sélecteur de vecteurs
garder = lambda vec1, vec2: sum(np.in1d(vec1, vec2)) == 0

def classifRangsEgaux(rangJoueur, matIndices, joueur, vecTable):
    n = matIndices.shape[1]
    resultat = np.zeros(n)
    combos = COMBOS.loc[COMBOS.rang==rangJoueur].combo
    carteHaute = combos.index[0] >= IND_CARTE_HAUTE
    p = 13 if carteHaute else 52    
    debut = IND_CARTE_HAUTE if carteHaute else 0
    if carteHaute:
        vecTable[np.where(vecTable)[0]%13] = 1
        vecTable[13:] = 0
    # On vérifie si l'égalité provient des cartes partagées
    matchTable = vecTable[:p].dot(MAT_COMBOS[debut:, :p].T)
    matchTable = np.where(matchTable >= LONGUEURS[debut:])[0] 
    egalParTable = False
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
        # matCas contient les indicatrices du joueur et des mains concurrentes qu'on évalue
        matCas = np.zeros((n+1)*p).reshape((n+1),p)
        matCas[np.repeat(np.arange(n+1),2), 
               np.concatenate([joueur, np.concatenate(matIndices.T)])%p] = 1
        matCas[:, np.where(vecTable)[0]%p] = 1
        # matCombosEg contient les indicatrices de toutes les combinaisons donnant lieu
        # à une égalité avec les mains évaluées
        matCombosEg = np.zeros(p*len(combos)).reshape(len(combos),p)
        matCombosEg[np.repeat(np.arange(len(combos)),len(combos.iloc[0])),
                    np.concatenate([c for c in combos])] = 1
        matCompar = matCas.dot(matCombosEg.T)   
        findMin = lambda vec: min(np.where(vec >= len(combos.iloc[0]))[0])
        # Pour chaque cas qu'on évalue, on remet à zéro les indicatrices des cartes inclues
        # dans la première combinaison avec laquelle notre cas match
        matCas -= matCombosEg[np.apply_along_axis(findMin, 1, matCompar)]
        matCas[matCas < 0] = 0
        # On crée une matrice dans laquelle toute valeur négative indiquera l'absence de valeur
        matCasOrd = -1 * np.ones(matCas.shape[0]*8).reshape(matCas.shape[0],8)
        for i in np.arange(matCas.shape[0]):
            vec = (np.sort(np.where(matCas[i])[0]%13))[::-1]
            matCasOrd[i,:len(vec)] = vec
        # On resépare les cartes du joueur et celles des adversaires
        joueur, matCasOrd = matCasOrd[0], matCasOrd[1:]
        resultat[np.where(matCasOrd[:,0] > joueur[0])[0]] = 1
        egaux = np.where(matCasOrd[:,0] == joueur[0])[0]
        for i in range(1,7):
            if len(egaux) == 0: break
            resultat[egaux[np.where(matCasOrd[egaux,i] > joueur[i])[0]]] = 1
            egaux = np.array([eg for eg in np.where(matCasOrd[:,i]==joueur[i])[0] if eg in egaux])
            if joueur[i+1] == -1 and len(egaux) > 0:
                resultat[egaux] = 0.5
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
    matPF[rAOb[0,selecteur[adv_gagne]], rAOb[1,selecteur[adv_gagne]]] += 1    
    # Traitement des égalités, s'il y a lieu
    adv_egal = np.where(rangsAdv == rangJoueur)[0]
    if len(adv_egal) > 0:
        vecRangsEgaux = classifRangsEgaux(rangJoueur, rAOb[:, selecteur[adv_egal]], 
                                          np.where(vecJoueur)[0], vecTable)
        matPF[rAOb[0,selecteur[adv_egal]], rAOb[1,selecteur[adv_egal]]] += vecRangsEgaux
    return matPF

def matInfluence(matrice):
    n = matrice.shape[0]
    matInfl = np.zeros(n*n).reshape(n,n)
    for i in range(0,n):
        matInfl[i,i] = matrice[i,i] - sum(matrice[i,:]) - sum(matrice[:,i])
    # Ajout des croisements; symmétrique pour simplifier l'accès ultérieur
    matInfl += np.triu(matrice,1) + np.triu(matrice,1).T
    return matInfl
    
def recursion(matInfl, base, denominateur, adv_restant, matPos, sepSplit, fixe=np.zeros(52)):
    k, total = int(sum(fixe)), base
    n = matPos.shape[0]
    if k > 0:
        # Ajustement du total de base
        f = np.where(fixe)[0]
        lignes = np.repeat(f,range(k,0,-1)).astype('int')
        cols = np.concatenate([f[i:k] for i in range(0,k)]).astype('int')
        #total += sum(matInfl[lignes, cols])
        total += sum(matInfl[lignes, cols])
    if adv_restant > 1:
        vecRes = np.zeros(n) # Vecteur de résultats de récursion
        denom_mod = (43-k)*(42-k)/2
        matFixe = np.repeat(fixe,n).reshape(52,n).T + matPos
        matFixe[matFixe>1] = 1
        matGarder = matPos.dot(matPos.T) == 0
        vecSepSplit = matGarder[:,:sepSplit].sum(1)
        for p in np.arange(n):
            # TEST!!! Si ça marche pas on rechange "base" pour "total"
            vecRes[p] = recursion(matInfl, base, denom_mod, adv_restant-1,
                                  matPos[matGarder[p]], vecSepSplit[p], matFixe[p])        
        total += sum(vecRes) - 0.5 * sum(vecRes[sepSplit:])
    return total / denominateur

def probaPlusFortTableFixe(vecJoueur, vecTable, n_adv):
    exclus = np.where(vecJoueur + vecTable)[0]
    matPF = matPlusFort(vecJoueur, vecTable, exclus)
    if n_adv > 1:
        matInfl = matInfluence(matPF)
        # Identification des positions dans la matrice liées à des combos d'adversaire
        # moins fortes que celle du joueur
        posMF = np.where(np.triu(np.ones(52*52).reshape(52,52) - 2 * matPF, 1) > 0)
        posMF = np.concatenate([posMF[0], posMF[1]]).reshape(2,len(posMF[0])).T
        if posMF.shape[0] > 0:
            posMF = posMF[np.apply_along_axis(garder,1,posMF,exclus)]
        # Identification des positions dans la matrice liées à des split pots
        posSplit = np.where(matPF==0.5)
        posSplit = np.concatenate([posSplit[0], posSplit[1]]).reshape(2,len(posSplit[0])).T
        if posSplit.shape[0] > 0:
            posSplit = posSplit[np.apply_along_axis(garder,1,posSplit,exclus)]
        # Réécriture sous forme matricielle, pour accélérer les calculs
        nMF = len(posMF)
        nTot = nMF + len(posSplit)
        matPos = np.zeros(52*nTot).reshape(nTot, 52)
        matPos[np.repeat(np.arange(nTot),2), 
               np.concatenate([posMF.flatten(), posSplit.flatten()])] = 1
        # Récursion sur les emplacements où est pas déjà certain que l'adversaire gagne
        proba = recursion(matInfl, matPF.sum(), 45*44/2, n_adv, matPos, nMF)
    else:
        proba = matPF.sum() / (45*44/2)
    return proba

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