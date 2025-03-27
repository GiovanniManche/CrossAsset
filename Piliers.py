import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
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
    


data = pd.read_excel("Inputs.xlsx", sheet_name= "Actions").set_index("Date")
cols_with_na = [col for col in data.columns if data[col].isnull().sum() != 0]
data[cols_with_na] = data[cols_with_na].interpolate(method = 'linear')

x = Pilier(data, "daily", "discret")

x.plot_cumulative_returns()



