import abc
import pandas as pd
import numpy as np
import math

class Strategy(metaclass = abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, returns: pd.DataFrame) -> None:
        self.returns: pd.DataFrame = returns
    
    @abc.abstractmethod
    def get_position(self) -> pd.DataFrame:
        pass


class Momentum(Strategy):
    """
    Classe qui implémente une stratégie Momentum basée sur les rendements des actifs. Deux solutions sont proposés : 
        - on classe les actifs qui montent "du plus fort" au moins fort / qui baissent, et on alloue des poids en fonction du rang ;
        - on prend les actifs qui montent et leur allouons un poids "équipondéré"
    
    Attributs : 
    ---
        returns : dataframe contenant les rendements considérés. C'est dans la classe Pilier, fonction "Ptf_construct" qu'on fera la boucle glissante sur 
            les rendements à considérer
        isEquiponderated : booléen indiquant si les parts sont calculées de manière équipondérée ou non. 
    ---  
    """
    def __init__(self, returns: pd.DataFrame, isEquiponderated : bool) -> None:
        self.returns: pd.DataFrame = returns
        self.isEquiponderated: bool = isEquiponderated

    def get_position(self) -> pd.DataFrame:
        """
        Calcule les parts dans le portefeuille à une date donnée. 
        """
        # Calcul du rendement sur la période (à vérifier si c'est bien ça qu'on utilise comme signal)
        signals_momentum: pd.Series = (1+ self.returns).prod() - 1

        # Initialisation des poids à 0
        weights: pd.DataFrame = pd.DataFrame(0.0, index=[self.returns.index[-1]], columns=self.returns.columns)
        if not self.isEquiponderated:
            # Ascending = True pour que les top performer aient le plus grand rank
            ranks: pd.Series = signals_momentum.rank(method="max", ascending=True).astype(float)
            weights.loc[weights.index[0], ranks.index] = ranks / ranks.sum()
       
        else:
            top_assets = signals_momentum[signals_momentum > 0].index
            if len(top_assets) > 0:
                weights.loc[weights.index[0], top_assets] = 1.0/len(top_assets)
            # Si tous les actifs sont dans le rouge, on passe en ranking
            else:
                ranks = signals_momentum.rank(method="max", ascending=True).astype(float)
                weights.loc[weights.index[0], ranks.index] = ranks / ranks.sum()

        return weights
            
