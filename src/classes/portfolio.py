import pandas as pd
import numpy as np
import datetime as datetime

from aiohttp.web_routedef import static
from dateutil.relativedelta import relativedelta
from operator import add

import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple

from classes.strategies import Strategy, Momentum, LowVol


class Portfolio:

    METHODS_LABELS = ["discret","continu"]

    DAILY_LABEL = "daily"
    WEEKLY_LABEL = "weekly"
    MONTHLY_LABEL = "monthly"
    QUARTERLY_LABEL = "quarterly"
    YEARLY_LABEL = "yearly"

    PERIODICITY_LABELS = [DAILY_LABEL, WEEKLY_LABEL, MONTHLY_LABEL, QUARTERLY_LABEL, YEARLY_LABEL]
    STRAT_LABELS = ["momentum", "low vol"]
    WEIGHTING_LABELS = ["equalweight", "ranking"]

    EQUITY_LABEL = "equity"
    BOND_LABEL = "bond"
    
    """
    Classe qui s'occupe des différentes métriques et de la composition des portefeuilles pour chaque pilier.
    Arguments : 
        - asset_prices : dataframe contenant les prix des actifs de l'univers considéré pour ce pilier, 
        - periodicity : période pour le calcul des rendements (daily, weekly, monthly, quarterly, yearly)
        - rebalancement : fréquence pour le calcul de la date de rebalancement
        - method : méthode pour le calcul des rendements (discret, continu)
        - list_strat : liste des stratégies à appliquer dans la poche d'alpha
        - list_weighting : liste des schémas de pondération à appliquer pour chaque stratégie de la poche d'alpha
        - calculation_window : taille de la fenêtre à utiliser pour les stratégies d'alpha
        - asset_class : classe d'actif pour laquelle on effectue le backtest (equity, bond, ...)
    """
    def __init__(self, data:pd.DataFrame, periodicity: str, rebalancement:str, method:str,
                 list_strat:list, list_weighting:list, calculation_window:int, asset_class:str):

        # Vérifications de cohérence
        if periodicity.lower() not in Portfolio.PERIODICITY_LABELS:
            raise Exception(f"La périodicité pour le calcul des rendements doit être l'une des suivantes : {Portfolio.PERIODICITY_LABELS}.")
        if method.lower() not in Portfolio.METHODS_LABELS:
            raise Exception(f"La méthode pour le calcul des rendements doit être l'une des suivantes : {Portfolio.METHODS_LABELS}.")
        if rebalancement.lower() not in Portfolio.PERIODICITY_LABELS:
            raise Exception(f"La périodicité pour la date de rebalancement doit être l'une des suivantes : {Portfolio.PERIODICITY_LABELS}.")
        if len(list_strat) == 0 or len(list_weighting) == 0:
            raise Exception("Les listes contenant les stratégies à mettre en oeuvre et les scémas de pondération associés "
                            "ne peuvent pas être vides")

        self.rebalancement: str = rebalancement
        self.periodicity: str = periodicity.lower()
        self.method: str = method.lower()
        self.calculation_window:int = calculation_window
        self.asset_class: str = asset_class

        self.benchmark: pd.DataFrame = self.get_bench(data)
        self.asset_prices:pd.DataFrame =  data.iloc[:,1:]
        self.log_asset_prices: pd.DataFrame = self.calc_log_prices() # Récupération des prix logarithmiques

        # Récupération des deux listes
        self.list_strategies:list = list_strat
        self.list_weighting:list = list_weighting

        # On calcule les rendements journaliers, hebdomadaires, mensuels, trimestriels, annuels
        self.returns: Dict[str, pd.DataFrame] = {period: self.compute_asset_returns(period) for period in Portfolio.PERIODICITY_LABELS}
        self.correlations = self.returns[self.periodicity].corr()

        # Calcul de l'allocation du portefeuille à chaque date
        self.positions:pd.DataFrame = pd.DataFrame()

        # Calcul de la NAV du portefeuille (= valeur du pilier)
        self.portfolio_value:pd.Series = pd.Series()


    def run_backtest(self):
        """
        Méthode permettant de réaliser le backtest
        """

        # 1ere étape : calcul des poids
        portfolio_positions: pd.DataFrame = self.build_portfolio()
        self.positions = portfolio_positions

        # 2eme étape : Calcul de la NAV du portefeuille
        portfolio_value: pd.Series = self.compute_portfolio_value()
        self.portfolio_value = portfolio_value


    def calc_log_prices(self):
        """
        Méthode permettant de convertir tous les prix en
        logarithmes pour la strat (permet d'atténuer les variations extrêmes)
        """
        return np.log10(self.asset_prices)

    def get_bench(self, data:pd.DataFrame):
        """
        Méthode permettant de récupérer le benchmark selon
        la périodicité souhaitée par l'utilisateur
        """
        period_resampling: dict = {
            "daily": None,
            "weekly": 'W-FRI',
            "monthly": 'ME',
            "quarterly": 'QE',
            "yearly": 'YE'
        }
        bench: pd.DataFrame = data.iloc[:, :1]
        resampled_bench: pd.DataFrame = bench if period_resampling[self.periodicity] is None else (
            bench.resample(period_resampling[self.periodicity]).last())
        return resampled_bench

    def rebalancing_date(self, prec_date:datetime = None) -> datetime:
        """
        Méthode permettant de calculer la nouvelle date de rebalancement
        à partir de la date de rebalancement précédente
        """

        # On rebalance à la première itération du backtest (=> à voir, peut équipondérer sinon)
        if self.rebalancement == Portfolio.DAILY_LABEL:
            next_rebalancing_date:datetime = prec_date.timedelta(days = 1)
        elif self.rebalancement == Portfolio.WEEKLY_LABEL:
            next_rebalancing_date:datetime = prec_date.timedelta(days = 7)
        elif self.rebalancement == Portfolio.MONTHLY_LABEL:
            next_rebalancing_date:datetime = prec_date + relativedelta(months=1)
        elif self.rebalancement == Portfolio.QUARTERLY_LABEL:
            next_rebalancing_date:datetime = prec_date + relativedelta(months = 3)
        elif self.rebalancement == Portfolio.YEARLY_LABEL:
            next_rebalancing_date:datetime = prec_date + relativedelta(years = 1)
        else:
            raise Exception("La fréquence de rebalancement souhaitée n'est pas implémentée")

        return next_rebalancing_date

    def compute_asset_returns(self, periodicity_returns:str) -> pd.DataFrame:
        """
        Calcule les rendements des actifs de l'univers en fonction de la périodicité et de la méthode
        """
        period_resampling: dict = {
            "daily": None,  
            "weekly": 'W-FRI',
            "monthly": 'ME',
            "quarterly": 'QE',
            "yearly": 'YE'
        }
        resampled_data: pd.DataFrame = self.asset_prices if period_resampling[periodicity_returns] is None else self.asset_prices.resample(period_resampling[periodicity_returns]).last()
        returns: pd.DataFrame = resampled_data.pct_change().dropna() if self.method == 'discret' else np.log(resampled_data).diff().dropna()
        return returns

    def build_portfolio(self) -> pd.DataFrame:
        """
        Méthode  pour calculer la valeur du portefeuille
        strategy : Momentum ou Low Vol
        calculation_window : 20 jours
        isEquiponderated : False (==> méthode de ranking utilisé)
        """

        # Voir calculation_window et periodicité. Pour moi par défaut c'est dans la même unité sinon étrange
        # Vérifier si la strat burn pas de périodes

        # On récupère le nombre de rendements disponibles
        length: int = self.returns[self.periodicity].shape[0]

        # Récupération de la première date de rebalancement (à modif : première à partir de laquelle la stratégie est applicable)
        rebalancing_date: datetime = self.returns[self.periodicity].index[self.calculation_window]
        # index_strat: int = self.index_rebalancing_date(rebalancing_date)
        index_strat:int = self.calculation_window

        # Initialisation du dataframe de positions / poids
        positions: pd.DataFrame = pd.DataFrame(0.0, index=self.returns[self.periodicity].index,
                                               columns=self.returns[self.periodicity].columns)

        # Boucle pour construire le portefeuille à partir de la première date de rebalancement
        for idx in range(index_strat, length):

            # Récupération de la date courante
            date: datetime = self.returns[self.periodicity].index[idx]

            # 1ere étape : Calcul des poids du portefeuille
            # 1er cas : date >= date de rebalancement ==> on rebalance
            if date >= rebalancing_date:

                # Récupération du premier indice pour le calcul des rendements
                begin_idx: int = idx - self.calculation_window

                # Calcule de la nouvelle date de rebalancement
                rebalancing_date = self.rebalancing_date(rebalancing_date)

                # Récupération des signaux associés à cette date selon les stratégies / schémas implémentés
                positions.iloc[idx, :] = self.compute_ptf_weight_div(self.returns[self.periodicity].iloc[begin_idx:idx, :], self.list_strategies,
                                                                 self.list_weighting)

            # 2e cas : la date n'est pas une date de rebalancement
            # Récupération des poids précédents et calcul des nouveaux poids
            else:
                prec_weights: list = positions.iloc[idx - 1, :]
                positions.iloc[idx, :] = self.compute_portfolio_derive(idx, prec_weights)

        # On ne conserve pas les dates inférieures avant la première date de rebalancement
        positions = positions.iloc[self.calculation_window:, ]
        return positions


    @staticmethod
    def compute_ptf_weight_div(returns_to_use:pd.DataFrame, list_strategies:list, list_weighting:list) -> list:
        """
        Méthode permettant de générer les poids d'un portefeuille
        à une date t à partir des poids générés par plusieurs stratégies.

        Hypothèse : chaque stratégie représente le même poids dans le portefeuille
        final (peut être optimisé)

        arguments :
        - list_strategies : liste contenant les stratégies à appliquer au sein du portefeuille
        - list_weighting : liste contenant les schémas de pondérations à appliquer au sein du portefeuille
        """
        # pondération de chaque stratégie dans le portefeuille
        scaling_factor:float = 1.0 / len(list_strategies)

        # Liste pour stocker les poids du portefeuille
        list_weights_ptf: list = [0] * returns_to_use.shape[1]

        # boucle sur les stratégies
        for i in range(len(list_strategies)):
            # récupération de la stratégie et du schéma de pondération associé
            strat:str = list_strategies[i]
            weighting:str = list_weighting[i]

            if strat not in Portfolio.STRAT_LABELS:
                raise Exception("Stratégie non implémentée")

            if weighting not in Portfolio.WEIGHTING_LABELS:
                raise Exception("Schéma de pondération non implémenté")

            # distinction selon les différentes stratégies implémentées
            if strat == Portfolio.STRAT_LABELS[0]:
                strategy_instance: Momentum = Momentum(returns_to_use,
                         weighting)
            # Ajouter un cas pour chaque stratégie
            elif strat == Portfolio.STRAT_LABELS[1]:
                strategy_instance: LowVol = LowVol(returns_to_use, weighting)
            else:
                raise Exception("To be implemented")
            
            list_weight_strat:list = strategy_instance.get_position()
            if len(list_weight_strat) != len(list_weights_ptf):
                raise Exception("Les séries de poids générés par les stratégies / le portefeuille doivent avoir la même longueur")

            # Mettre à jour les poids au sein du portefeuille
            list_weight_strat = [weight * scaling_factor for weight in list_weight_strat]
            list_weights_ptf = list(map(add, list_weights_ptf, list_weight_strat))

        return list_weights_ptf

    def compute_portfolio_derive(self, idx:int, list_prec_weights:list) -> list:
        """
        Méthode permettant de calculer la dérive des poids
        pour un portefeuille Long
        """

        if len(list_prec_weights) != len(self.returns[self.periodicity].iloc[idx - 1,:]):
            raise Exception("Les listes de poids et de rendement doivent avoir la même taille pour réaliser le calcul")

        # Liste pour stocker les nouveaux poids
        weights:list = []

        # Etape 1 : récupération des rendements à la période précédente
        list_prec_ret:list = self.returns[self.periodicity].iloc[idx - 1,:]

        # Etape 2 : calcul du rendement du portefeuille à la fin de la période précédente
        ptf_ret: float = np.sum(np.multiply(list_prec_weights, list_prec_ret))
        tot_ret: float = 1 + ptf_ret

        # Etape 3 : calcul des poids
        for i in range(len(list_prec_weights)):
            old_weight:float = list_prec_weights[i]
            tot_ret_asset: float = 1 + list_prec_ret[i]
            weights.append(old_weight * tot_ret_asset / tot_ret)

        return weights

    def compute_portfolio_value(self, initial_value:float = 100.0) -> pd.Series:
        # Vérifications élémentaires
        if self.returns[self.periodicity].isna().any().any():
            raise ValueError("Les données contiennent des NaN. Vérifiez les entrées dans `self.data`.")
        if self.positions.isna().any().any():
            raise ValueError("Les positions contiennent des NaN. Vérifiez les entrées dans `positions`.")

        portfolio_value = pd.Series(index=self.positions.index, dtype=float)
        portfolio_value.iloc[0] = initial_value  # La valeur initiale du portefeuille

        for t in range(1, len(self.positions)):  # On commence à t=1 car nous avons besoin des rendements de t

            # récupération de l'indice pour les rendements (rendement en t-1 pour la NAV en t)
            ret_idx:int = self.calculation_window + t

            # Si toutes les positions sont nulles, on conserve la valeur précédente (initiale)
            if self.positions.iloc[t - 1].all() == 0:
                portfolio_value.iloc[t] = portfolio_value.iloc[t - 1]
            else:
                asset_returns = self.returns[self.periodicity].iloc[ret_idx]
                weighted_returns = (self.positions.iloc[t - 1] * asset_returns).sum()
                portfolio_value.iloc[t] = portfolio_value.iloc[t - 1] * (1 + weighted_returns)

        # Mise à jour des valeurs de marché
        return portfolio_value



    
    