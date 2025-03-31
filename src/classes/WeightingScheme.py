from abc import ABC, abstractmethod
from enum import Enum
import pandas as pd

class Weighting_scheme_type(Enum):
    EQUAL_WEIGHT = "Equal Weight"
    MINVAR = "Min Variance"
    SHARPE = "Sharpe Ratio"
    RANK = "Ranking"

"""
Classe non fonctionnelle, implémentée dans strategy directement
"""

class WeightingScheme(ABC):
    @abstractmethod
    def compute_weights(self, signals: pd.Series):
        pass

class EquallyWeighting(WeightingScheme):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_weights(self, signals: pd.Series):
        """
        Méthode permettant de construire un portefeuille équipondéré
        Return : liste de poids associés à chaque actif du portefeuille
        """

        # Initialisation de la liste contenant les poids du portefeuille
        weights_ptf: list = []

        # Récupération
        nb_buy_signals: int = signals.loc[signals > 0].shape[0]
        if nb_buy_signals==0:
            ranking_instance: RankingWeightingSignals = RankingWeightingSignals()
            weights_ptf: list = ranking_instance.compute_weights(signals)
            return weights_ptf

        weight_long: float = 1 / nb_buy_signals

        for i in range(len(signals)):
            # Premier cas : portefeuille Long Only
                if signals[i] > 0:
                    weights_ptf.append(weight_long)
                else:
                    weights_ptf.append(0)

        return weights_ptf

class RankingWeightingSignals(WeightingScheme):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_weights(self, signals: pd.Series):
        """
        Méthode permettant de construire un portefeuille en adoptant une méthodologie de ranking
        list_signals : liste contenant les valeurs des signaux renvoyés par la stratégie
        type_
        """
        weights_ptf:list = []
        ranks: pd.Series = signals.rank(method="max", ascending=True).astype(float)
        for i in range(len(ranks)):
            weights_ptf.append(ranks[i] / ranks.sum())
        return weights_ptf