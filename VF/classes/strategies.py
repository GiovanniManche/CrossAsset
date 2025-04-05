import abc
import pandas as pd
import numpy as np

from classes.weightingScheme import EquallyWeighting, RankingWeightingSignals

class Strategy(metaclass=abc.ABCMeta):
    EQUALWEIGHT_LABEL = "equalweight"
    RANKIG_LABEL = "ranking"

    @abc.abstractmethod
    def __init__(self, returns: pd.DataFrame) -> None:
        self.returns: pd.DataFrame = returns

    @abc.abstractmethod
    def get_position(self) -> list:
        pass


"""
classe Momentum
"""
class Momentum(Strategy):
    """
    Classe qui implémente une stratégie Momentum basée sur les rendements des actifs. Deux solutions sont proposées :
        - on classe les actifs qui montent "du plus fort" au moins fort / qui baissent, et on alloue des poids en fonction du rang ;
        - on prend les actifs qui montent et leur allouons un poids "équipondéré"

    Attributs :
    ---
        returns : dataframe contenant les rendements considérés.
        isEquiponderated : booléen indiquant si les parts sont calculées de manière équipondérée ou non.
    ---
    """

    def __init__(self, returns: pd.DataFrame, weight_scheme:str) -> None:
        self.returns: pd.DataFrame = returns
        self.weight_scheme: str = weight_scheme

    def get_position(self) -> list:
        """
        Calcule les parts dans le portefeuille à une date donnée.
        """
        # Calcul du rendement sur la période (à vérifier si c'est bien ça qu'on utilise comme signal)
        signals_momentum: pd.Series = (1 + self.returns).prod() - 1

        # Cas où l'utilisateur souhaite réaliser une allocation par ranking
        if self.weight_scheme == Strategy.RANKIG_LABEL:
            ranking_instance: RankingWeightingSignals = RankingWeightingSignals()
            weights: list = ranking_instance.compute_weights(signals_momentum)

        elif self.weight_scheme == Strategy.EQUALWEIGHT_LABEL:
            equalweight_instance:EquallyWeighting = EquallyWeighting()
            weights: list = equalweight_instance.compute_weights(signals_momentum)
        else:
            raise Exception("Méthode non implémentée")

        check_weight = np.sum(weights)
        if round(check_weight,5) != 1:
            raise Exception("Erreur dans le calcul des poids. La somme des poids doit être égale à 1")
        return weights

class LowVol(Strategy):
    """
    Classe qui implémente une stratégie Low Vol basée sur la volatilité des actifs.
    Deux solutions sont proposées :
        - on classe les actifs du "moins volatile" au "plus volatile", et on alloue des poids en fonction du rang ;
        - on prend les actifs les moins volatiles et leur allouons un poids "équipondéré"
    
    Attributs :
    ---
        returns : dataframe contenant les rendements considérés pour calculer la volatilité.
        weight_scheme : chaîne de caractères indiquant le schéma de pondération à utiliser ('ranking' ou 'equalweight').
    ---
    """
    
    def __init__(self, returns: pd.DataFrame, weight_scheme: str) -> None:
        self.returns: pd.DataFrame = returns
        self.weight_scheme: str = weight_scheme
    
    def get_position(self) -> list:
        """
        Calcule les parts dans le portefeuille à une date donnée.
        """
        # Calcul de la volatilité sur la période considérée
        signals_low_vol: pd.Series = self.returns.std()
        
        # Comme on veut favoriser les actifs à faible volatilité, on inverse le signal
        signals_low_vol: pd.Series = 1 / signals_low_vol
        
        # Cas où l'utilisateur souhaite réaliser une allocation par ranking
        if self.weight_scheme == Strategy.RANKIG_LABEL:
            ranking_instance: RankingWeightingSignals = RankingWeightingSignals()
            weights: list = ranking_instance.compute_weights(signals_low_vol)
        
        # Cas où l'utilisateur souhaite réaliser une allocation équipondérée
        elif self.weight_scheme == Strategy.EQUALWEIGHT_LABEL: 
            equalweight_instance: EquallyWeighting = EquallyWeighting()
            weights: list = equalweight_instance.compute_weights(signals_low_vol)
        else:
            raise Exception("Méthode non implémentée")
        
        # Vérification que la somme des poids est égale à 1
        check_weight = np.sum(weights)
        if round(check_weight, 5) != 1:
            raise Exception("Erreur dans le calcul des poids. La somme des poids doit être égale à 1")
            
        return weights
