import pandas as pd
import numpy as np
import datetime as datetime
from dateutil.relativedelta import relativedelta
from operator import add

import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple

from src.classes.Strategies import Strategy, Momentum, Momentum2


class Portfolio:

    METHODS_LABELS = ["discret","continu"]

    DAILY_LABEL = "daily"
    WEEKLY_LABEL = "weekly"
    MONTHLY_LABEL = "monthly"
    QUARTERLY_LABEL = "quarterly"
    YEARLY_LABEL = "yearly"

    PERIODICITY_LABELS = [DAILY_LABEL, WEEKLY_LABEL, MONTHLY_LABEL, QUARTERLY_LABEL, YEARLY_LABEL]
    STRAT_LABELS = ["momentum", "low Vol"]
    WEIGHTING_LABELS = ["equalweight", "ranking"]
    
    """
    Classe qui s'occupe des différentes métriques et de la composition des portefeuilles pour chaque pilier.
    Arguments : 
        - asset_prices : dataframe contenant les prix des actifs de l'univers considéré pour ce pilier, 
        - periodicity : période pour le calcul des rendements (daily, weekly, monthly, quarterly, yearly)
        - rebalancement : fréquence pour le calcul de la date de rebalancement
        - method : méthode pour le calcul des rendements (discret, continu)
    """
    def __init__(self, data:pd.DataFrame, periodicity: str, rebalancement:str, method:str,
                 list_strat:list, list_weighting:list, calculation_window:int):

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

        self.benchmark: pd.DataFrame = data.iloc[:, :1]
        self.asset_prices:pd.DataFrame =  data.iloc[:,1:]
        self.log_asset_prices: pd.DataFrame = self.calc_log_prices() # Récupération des prix logarithmiques

        self.rebalancement: str = rebalancement
        self.periodicity: str = periodicity.lower()
        self.method: str = method.lower()
        self.calculation_window:int = calculation_window

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
        portfolio_positions: pd.DataFrame = self.build_portfolio_v3()
        self.positions = portfolio_positions

        # 2eme étape : Calcul de la NAV du portefeuille
        portfolio_value: pd.Series = self.compute_portfolio_value_v2()
        self.portfolio_value = portfolio_value


    """
    à voir si pertinent de travailler en
    prig logarithmiques pour certaines strat plutôt qu'en prix
    """
    def calc_log_prices(self):
        """
        Méthode permettant de convertir tous les prix en
        logarithmes pour la strat (permet d'atténuer les variations extrêmes)
        """
        return np.log10(self.asset_prices)

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

    def build_portfolio_v3(self) -> pd.DataFrame:
        """
        Méthode  pour calculer la valeur du portefeuille
        strategy : Momentum
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
                positions.iloc[idx, :] = self.compute_ptf_weight(self.returns[self.periodicity].iloc[begin_idx:idx, :], self.list_strategies,
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
    def compute_ptf_weight(returns_to_use:pd.DataFrame, list_strategies:list, list_weighting:list) -> list:
        """
        Méthode permettant de générer les poids d'un portefeuille
        à une date t à partir des poids générés par plusieurs stratégies.

        Hypothèse : chaque stratégie représente le même poids dans le portefeuille
        final (peut être optimisé)

        arguments :
        - list_strategies : liste contenant les stratégies à appliquer au sein du portefeuille
        - list_weighting : liste contenant les schémas de pondérations à appliquer au sein du portefeuille
        """
        if len(list_strategies) != len(list_weighting):
            raise Exception("Chaque stratégie à appliquer doit avoir un schéma de pondération associé")

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
                strategy_instance: Momentum2 = Momentum2(returns_to_use,
                         weighting)
                list_weight_strat:list = strategy_instance.get_position()

            # Ajouter un cas pour chaque stratégie
            else:
                raise Exception("To be implemented")

            if len(list_weight_strat) != len(list_weights_ptf):
                raise Exception("Les séries de poids générés par les stratégies / le portefeuille doivent avoir la même longueur")

            # Mettre à jour les poids au sein du portefeuille
            list_weight_strat = [weight * scaling_factor for weight in list_weight_strat]
            list_weights_ptf = list(map(add, list_weights_ptf, list_weight_strat))

        return list_weights_ptf

    """
    a discuter avec le prof
    """
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

    def compute_portfolio_value_v2(self, initial_value:float = 100.0) -> pd.Series:
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

    def heatmap_correlations(self, annot: bool = True, cmap: str = 'coolwarm', vmin: float = -1,
                             vmax: float = 1) -> None:
        """
        Affiche la matrice de corrélation des rendements (pour la périodicité donnée à l'instanciation)
        Arguments :
            - l'instance
            - annot : booléen qui vaut True si l'on souhaite que les valeurs des corrélations soient affichées dans les cases de la heatmap,
            - cmap : str pour les couleurs de la heatmap,
            - vmin, vmax : réglent l'échelle de la grille
        """
        plt.figure(figsize=(12, 6))

        heatmap = sns.heatmap(
            self.correlations,
            annot=annot,
            fmt=".2f",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
        )
        plt.title(f'Matrice de corrélation - rendements {self.periodicity}', fontsize=16, pad=20)
        plt.tight_layout()
        plt.show()

    def plot_cumulative_returns(self) -> None:
        """
        Calcule et affiche les rendements cumulés pour chaque actif qui compose l'univers.
        """
        cumulative_returns: pd.DataFrame = (1 + self.returns[self.periodicity]).cumprod()
        cumulative_returns.plot()
        plt.title(f"Rendements cumulés (rendements {self.periodicity})")
        plt.xlabel("Date")
        plt.ylabel("Valeur")
        plt.legend(title="Actifs", frameon=True, fontsize=8)
        plt.axhline(y=1, color="red", linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.show()
    """
    old
    """

    def build_portfolio_v2(self, strategy: str = "Momentum", calculation_window: int = 20,
                           isEquiponderated: bool = False) -> pd.DataFrame:
        """
        Méthode  pour calculer la valeur du portefeuille
        strategy : Momentum
        calculation_window : 20 jours
        isEquiponderated : False (==> méthode de ranking utilisé)
        """

        # Voir calculation_window et periodicité. Pour moi par défaut c'est dans la même unité sinon étrange
        # Vérifier si la strat burn pas de périodes

        # On récupère le nombre de rendements disponibles
        length: int = self.returns[self.periodicity].shape[0]

        # Récupération de la première date de rebalancement (à modif : première à partir de laquelle la stratégie est applicable)
        rebalancing_date: datetime = self.returns[self.periodicity].index[calculation_window]
        index_strat: int = self.index_rebalancing_date(rebalancing_date)
        # index_strat:int = calculation_window

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
                begin_idx: int = idx - calculation_window

                # Calcule de la nouvelle date de rebalancement
                rebalancing_date = self.rebalancing_date(rebalancing_date)

                # Récupération des signaux associés à cette date
                if strategy.lower() == "momentum":
                    strategy_instance = Momentum(self.returns[self.periodicity].iloc[begin_idx:idx, :],
                                                 isEquiponderated)
                    positions.iloc[idx, :] = strategy_instance.get_position()

                else:
                    raise Exception("Stratégie non implémentée pour le moment")

            # 2e cas : la date n'est pas une date de rebalancement
            # Récupération des poids précédents et calcul des nouveaux poids
            else:
                prec_weights: list = positions.iloc[idx - 1, :]
                positions.iloc[idx, :] = self.compute_portfolio_derive(idx, prec_weights)

        # On ne conserve pas les dates inférieures avant la première date de rebalancement
        positions = positions.iloc[calculation_window:, ]
        self.positions = positions
        return positions

    def build_portfolio(self, strategy: str = "Momentum", calculation_window: int = 20, rebalancing_frequency: int = 1, isEquiponderated: bool = False) -> pd.DataFrame:
        """
        Calcul des positions de la stratégie. Le fonctionnement est le suivant : 
            - On met à jour le jeu de données passé en paramètre de la fonction "get_position" de la stratégie en fonction
                de la fréquence de rebalancement et de la fenêtre de calcul. Par exemple, si on veut du Momentum sur les rendements
                des 12 derniers mois, avec rebalancement mensuel et qu'on a une périodicité mensuelle, on va récupérer les rendements 
                des 12 derniers mois et les passer en argument. 
            - On récupère, à chaque date de rebalancement, les positions ainsi calculées (entre deux dates de rebalancement, les positions restent les mêmes).
        Le résultat renvoyé est un dataframe contenant les positions par actif à chaque date. 
        """
        # On initialise le dataframe de positions
        positions: pd.DataFrame = pd.DataFrame(0.0, index = self.returns[self.periodicity].index, columns= self.returns[self.periodicity].columns)


        # On récupère le nombre de rendements disponibles
        length: int = self.returns[self.periodicity].shape[0]
        # Boucle : on commence a minima au nombre de rendement nécessaire pour la fenêtre de calcul, et on avance ensuite 
        # par pas égal à la fréquence de rebalancement
        for idx in range(calculation_window, length, rebalancing_frequency):
            # Le premier index = idx - fenêtre de calcul (pour avoir, par exemple, les rendements des 12 DERNIERS mois)
            begin_idx = idx - calculation_window
            # Fréquence de rebalancement : si on n'a pas assez de rendements à la fin, on ne prend que les rendements restants
            correct_frequency = rebalancing_frequency  if (length - idx > rebalancing_frequency) else (length - idx)
            # Ajout des positions
            if strategy.lower() == "momentum":
                strategy_instance = Momentum(self.returns[self.periodicity].iloc[begin_idx:idx,:], isEquiponderated)

            # Attention : valeur pas constante avec cette méthode, cf excel, à voir
            positions.iloc[idx: idx + correct_frequency, :] = strategy_instance.get_position()
        return positions

    def compute_portfolio_value(self, positions: pd.DataFrame, initial_value: float = 100.0) -> pd.Series:
        # Vérifications élémentaires
        if not positions.index.equals(self.returns[self.periodicity].index) or not positions.columns.equals(self.returns[self.periodicity].columns):
            raise ValueError("Les indices ou colonnes de `positions` et `data` ne correspondent pas.")
        if self.returns[self.periodicity].isna().any().any():
            raise ValueError("Les données contiennent des NaN. Vérifiez les entrées dans `self.data`.")
        if positions.isna().any().any():
            raise ValueError("Les positions contiennent des NaN. Vérifiez les entrées dans `positions`.")
    
        portfolio_value = pd.Series(index=self.returns[self.periodicity].index, dtype=float)
        portfolio_value.iloc[0] = initial_value  # La valeur initiale du portefeuille
    
        for t in range(1, len(positions)):  # On commence à t=1 car nous avons besoin des rendements de t
            # Si toutes les positions sont nulles, on conserve la valeur précédente (initiale)
            if positions.iloc[t-1].all() == 0:
                portfolio_value.iloc[t] = portfolio_value.iloc[t-1]
            else:
                asset_returns = self.returns[self.periodicity].iloc[t]
                weighted_returns = (positions.iloc[t-1] * asset_returns).sum()
                portfolio_value.iloc[t] = portfolio_value.iloc[t-1] * (1 + weighted_returns)
    
        # Mise à jour des valeurs de marché
        self.portfolio_value = portfolio_value
        return portfolio_value

    def index_rebalancing_date(self, first_rebalancing_date: datetime) -> int:
        """
        Méthode permettant de déterminer
        l'indice de la première date de rebalancement (= indice
        à partir duquel on commence la boucle pour le backtester)
        """

        # Récupération de la liste des dates
        liste_dates : list = self.returns[self.periodicity].index.to_list()
        for t in range(len(liste_dates)-1):
            date:datetime = liste_dates[t]
            # Dès qu'une date est supérieure ou égale à la date de rebalancement, on la récupère
            if date>=first_rebalancing_date:
                return t

    

"""
data = pd.read_excel("data/Données.xlsx", sheet_name= "Actions").set_index("Date")
cols_with_na = [col for col in data.columns if data[col].isnull().sum() != 0]
data[cols_with_na] = data[cols_with_na].interpolate(method = 'linear')

x = Pilier(data, "monthly", "monthly", "discret")

x.plot_cumulative_returns()
pos = x.build_portfolio()
pos.to_excel("test positions Momentum.xlsx")
x.compute_portfolio_value(pos, 100).to_excel("Ptf value.xlsx")
"""