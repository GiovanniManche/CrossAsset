import abc
import pandas as pd
import numpy as np

from src.classes.WeightingScheme import EquallyWeighting, RankingWeightingSignals

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
Classe Action Coeur de portefeuille
"""
class CoreEquity(Strategy):
    EUROSTOXX_LABEL = "SX5E Index"
    SP500_INDEX = "SPX Index"
    EMERGING_INDEX = "MXEF Index"
    """
    Classe qui permet de construire l'allocation
    coeur de portefeuille sur le segment equity
    """
    def __init__(self, returns: pd.DataFrame, *args, **kwargs) -> None:
        self.returns: pd.DataFrame = returns

    def get_position(self) -> list:
        if (CoreEquity.EUROSTOXX_LABEL not in self.returns.columns.values or
                CoreEquity.SP500_INDEX not in self.returns.columns.values or
                CoreEquity.EMERGING_INDEX not in self.returns.columns.values):
            raise Exception("L'indice Eurostoxx ne figure pas dans l'univers action. Il faut modifier l'indice à "
                            "utiliser pour la stratégie Large Cap Core Euro")

        weights: list = [0] * self.returns.shape[1]
        index_eurostoxx:int = self.returns.columns.get_loc(CoreEquity.EUROSTOXX_LABEL)
        index_spx: int = self.returns.columns.get_loc(CoreEquity.SP500_INDEX)
        index_emerging: int = self.returns.columns.get_loc(CoreEquity.EMERGING_INDEX)

        # Récupération des poids pour chacun de ces indices : 1/3, 1/3, 1/3
        weights[index_eurostoxx] = 1.0/3.0
        weights[index_spx] = 1.0/3.0
        weights[index_emerging] = 1.0/3.0
        return weights

"""
Classe Obligation Coeur de portefeuille
"""
class CoreBond(Strategy):

    IG_LABEL = "LEGATRUU Index"
    TREASURY_INDEX = ""

    """
    Classe qui permet de construire l'allocation
    coeur de portefeuille sur le segment obligataire
    """
    def __init__(self, returns: pd.DataFrame, *args, **kwargs) -> None:
        self.returns: pd.DataFrame = returns

"""
classe Momentum vTim
"""
class Momentum2(Strategy):
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


"""
Old
"""
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

    def __init__(self, returns: pd.DataFrame, isEquiponderated: bool) -> None:
        self.returns: pd.DataFrame = returns
        self.isEquiponderated: bool = isEquiponderated

    def get_position(self) -> list:
        """
        Calcule les parts dans le portefeuille à une date donnée.
        """
        # Calcul du rendement sur la période (à vérifier si c'est bien ça qu'on utilise comme signal)
        signals_momentum: pd.Series = (1 + self.returns).prod() - 1

        # Initialisation des poids à 0
        weights: pd.DataFrame = pd.DataFrame(0.0, index=[self.returns.index[-1]], columns=self.returns.columns)
        if not self.isEquiponderated:
            # Ascending = True pour que les top performer aient le plus grand rank
            ranks: pd.Series = signals_momentum.rank(method="max", ascending=True).astype(float)
            weights.loc[weights.index[0], ranks.index] = ranks / ranks.sum()

        else:
            top_assets = signals_momentum[signals_momentum > 0].index
            if len(top_assets) > 0:
                weights.loc[weights.index[0], top_assets] = 1.0 / len(top_assets)
            # Si tous les actifs sont dans le rouge, on passe en ranking
            else:
                ranks = signals_momentum.rank(method="max", ascending=True).astype(float)
                weights.loc[weights.index[0], ranks.index] = ranks / ranks.sum()
        test = sum(weights.values.flatten().tolist())
        if round(test,5) != 1:
            raise Exception("Erreur dans le calcul des poids. La somme des poids doit être égale à 1")
        return weights.values.flatten().tolist()

"""
Classe Action Core Europe
"""
class CoreEquityEU(Strategy):
    EUROSTOXX_LABEL = "SX5E Index"
    """
    Classe qui permet de construire l'allocation coeur
    sur le segment actions large cap Européen
    """
    def __init__(self, returns: pd.DataFrame, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.returns: pd.DataFrame = returns

    def get_position(self) -> list:
        """
        Exposition pur à un indice large cap euro : Eurostoxx 50
        """
        weights: list = [0] * self.returns.shape[1]
        if CoreEquityEU.EUROSTOXX_LABEL not in self.returns.columns.values:
            raise Exception("L'indice Eurostoxx ne figure pas dans l'univers action. Il faut modifier l'indice à "
                            "utiliser pour la stratégie Large Cap Core Euro")
        index_eurostoxx:int = self.returns.columns.get_loc(CoreEquityEU.EUROSTOXX_LABEL)
        weights[index_eurostoxx] = 1
        return weights

"""
Classe Action Core US
"""
class CoreEquityUS(Strategy):
    SP500_INDEX = "SPX Index"
    """
    Classe qui permet de construire l'allocation coeur
    sur le segment actions large cap US
    """
    def __init__(self, returns: pd.DataFrame, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.returns: pd.DataFrame = returns

    def get_position(self) -> list:
        """
        Exposition pur à un indice large cap euro : Eurostoxx 50
        """
        weights: list = [0] * self.returns.shape[1]
        if CoreEquityUS.SP500_INDEX not in self.returns.columns.values:
            raise Exception("L'indice S&P 500 ne figure pas dans l'univers action. Il faut modifier l'indice à "
                            "utiliser pour la stratégie Large Cap Core US")
        index_sp:int = self.returns.columns.get_loc(CoreEquityUS.SP500_INDEX)
        weights[index_sp] = 1
        return weights

