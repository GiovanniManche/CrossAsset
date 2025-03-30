from abc import ABC, abstractmethod
from enum import Enum

from pyarrow import list_


class Weighting_scheme_type(Enum):
    EQUAL_WEIGHT = "Equal Weight"
    MINVAR = "Min Variance"
    SHARPE = "Sharpe Ratio"
    RANK = "Ranking"

class WeightingScheme(ABC):
    @abstractmethod
    def compute_weights(self, list_signals: list, type_ptf:str):
        pass

class EquallyWeighting(WeightingScheme):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_weights(self, list_signals: list, type_ptf:str):
        """
        Méthode permettant de construire un portefeuille équipondéré
        Return : liste de poids associés à chaque actif du portefeuille
        """

        # Initialisation de la liste contenant les poids du portefeuille
        weights_ptf: list = []

        # Récupération
        nb_buy_signals: int = list_signals[list_signals == 1].Count()
        nb_sell_signals: int = list_signals[list_signals == -1].Count()

        weight_long: float = 1 / nb_buy_signals
        weight_short: float = 1 / nb_sell_signals

        for i in range(len(list_signals)-1):
            # Premier cas : portefeuille Long Only
            if type_ptf == Ptf_type.LONG:
                if list_signals[i] == 1:
                    weights_ptf.append(weight_long)
                else:
                    weights_ptf.append(0)

            # Deuxièmer cas : portefeuille short
            elif type_ptf == Ptf_type.SHORT:
                if list_signals[i] == -1:
                    weights_ptf.append(weight_short)
                else:
                    weights_ptf.append(0)

        # Troisième cas : Portefeuille Long Short
            elif type_ptf == Ptf_type.LS:
                if list_signals[i] == 1:
                    weights_ptf.append(weight_long)
                elif list_signals[i] == -1:
                    weights_ptf.append(-weight_short)
                else:
                    weights_ptf.append(0)
        # Autres cas
            else:
                raise Exception("La méthode n'est pas implémentée pour un autre type de portefeuille")
        return weights_ptf

