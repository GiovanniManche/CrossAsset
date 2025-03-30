import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple
from strategies import Strategy, Momentum


class Pilier:

    METHODS_LABELS = ["discret","continu"]
    PERIODICITY_LABELS = ["daily","weekly","monthly","quarterly","yearly"]
    
    """
    Classe qui s'occupe des différentes métriques et de la composition des portefeuilles pour chaque pilier.
    Arguments : 
        - asset_prices : dataframe contenant les prix des actifs de l'univers considéré pour ce pilier, 
        - periodicity : période pour le calcul des rendements (daily, weekly, monthly, quarterly, yearly)
        - method : méthode pour le calcul des rendements (discret, continu)
    """
    def __init__(self, asset_prices: pd.DataFrame, periodicity: str, method:str):
        # Vérifications de cohérence
        if periodicity.lower() not in Pilier.PERIODICITY_LABELS:
            raise Exception(f"La périodicité pour le calcul des rendements doit être l'une des suivantes : {Pilier.PERIODICITY_LABELS}.")
        if method.lower() not in Pilier.METHODS_LABELS:
            raise Exception(f"La méthode pour le calcul des rendements doit être l'une des suivantes : {Pilier.METHODS_LABELS}.")
        
        self.asset_prices: pd.DataFrame = asset_prices
        self.periodicity: str = periodicity.lower()
        self.method: str = method.lower()
        # On calcule les rendements journaliers, hebdomadaires, mensuels, trimestriels, annuels
        self.returns: Dict[pd.DataFrame] = {period: self.compute_asset_returns(period) for period in Pilier.PERIODICITY_LABELS}
        self.correlations = self.returns[self.periodicity].corr()

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
    
    def build_portfolio(self, strategy: str = "Momentum", calculation_window: int = 12, rebalancing_frequency: int = 1, isEquiponderated: bool = False) -> pd.DataFrame:
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
            if (positions.iloc[t-1] == 0).all():
                portfolio_value.iloc[t] = portfolio_value.iloc[t-1]
            else:
                asset_returns = self.returns[self.periodicity].iloc[t]
                weighted_returns = (positions.iloc[t-1] * asset_returns).sum()
                portfolio_value.iloc[t] = portfolio_value.iloc[t-1] * (1 + weighted_returns)
    
        # Mise à jour des valeurs de marché
        self.portfolio_value = portfolio_value
        return portfolio_value


    def heatmap_correlations(self, annot: bool =True, cmap: str='coolwarm', vmin: float =-1, vmax: float =1) -> None:
        """
        Affiche la matrice de corrélation des rendements (pour la périodicité donnée à l'instanciation)
        Arguments : 
            - l'instance
            - annot : booléen qui vaut True si l'on souhaite que les valeurs des corrélations soient affichées dans les cases de la heatmap, 
            - cmap : str pour les couleurs de la heatmap,
            - vmin, vmax : réglent l'échelle de la grille
        """
        plt.figure(figsize=(12,6))

        heatmap = sns.heatmap(
            self.correlations, 
            annot=annot,
            fmt =".2f",
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
        cumulative_returns: pd.DataFrame = (1+self.returns[self.periodicity]).cumprod()
        cumulative_returns.plot()
        plt.title(f"Rendements cumulés (rendements {self.periodicity})")
        plt.xlabel("Date")
        plt.ylabel("Valeur")
        plt.legend(title = "Actifs", frameon = True, fontsize = 8)
        plt.axhline(y=1, color = "red", linestyle = "--", alpha = 0.3)
        plt.tight_layout()
        plt.show()
    

"""
data = pd.read_excel("Inputs.xlsx", sheet_name= "Actions").set_index("Date")
cols_with_na = [col for col in data.columns if data[col].isnull().sum() != 0]
data[cols_with_na] = data[cols_with_na].interpolate(method = 'linear')

x = Pilier(data, "monthly", "discret")

x.plot_cumulative_returns()
pos = x.build_portfolio()
pos.to_excel("test positions Momentum.xlsx")
x.compute_portfolio_value(pos, 100).to_excel("Ptf value.xlsx")
test = x.asset_prices
print(x.compute_asset_returns("monthly"))"
"""