import math

import numpy as np
import pandas as pd

class Metrics():

    DISCRET_LABEL = "discret"
    CONTINU_LABEL = "continu"
    DAILY_LABEL = "Daily"
    WEEKLY_LABEL = "Weekly"
    MONTHLY_LABEL = "Monthly"
    QUARTERLY_LABEL = "Quarterly"
    YEARLY_LABEL = "Yearly"

    """
    Classe permettant de calculer un ensemble de métriques de performance et de risque pour un portefeuille
    Input : 
        - ptf_nav, une liste contenant l'évolution de la NAV du portefeuille dans le temps
        - method : méthode pour calculer les rendements
    """
    def __init__(self, ptf_nav:pd.DataFrame, method:str, periodicity:str):
        self.nav: pd.DataFrame = ptf_nav
        self.method: str = method
        self.periodicity: str = periodicity
        self.annualization_factor: int = self.compute_annualization_factor()
        self.returns: pd.DataFrame = self.compute_returns()

    def compute_annualization_factor(self):
        """
        Méthode permettant de déterminer le coefficient d'annualisation
        à utiliser selon la périodicité des données
        """
        if self.periodicity == Metrics.DAILY_LABEL:
            return 252
        elif self.periodicity == Metrics.WEEKLY_LABEL:
            return 52
        elif self.periodicity == Metrics.MONTHLY_LABEL:
            return 12
        elif self.periodicity == Metrics.QUARTERLY_LABEL:
            return 4
        elif self.periodicity == Metrics.YEARLY_LABEL:
            return 1
        else:
            raise Exception("Les calculs pour une périodicité autre ne sont pas implémentés")

    def compute_returns(self):
        """
        Méthode permettant de calculer les rendements du portefeuille
        selon la méthode spécifiée
        """
        if self.method != Metrics.DISCRET_LABEL and self.method != Metrics.CONTINU_LABEL:
            raise Exception("La méthode de calcul des rendements demandée n'est pas implémentée")

        returns:pd.DataFrame = pd.DataFrame()
        if self.method == Metrics.DISCRET_LABEL:
            returns = self.nav.pct_change()

        else:
            returns = np.log10(self.nav).diff()

        returns = returns[1:]
        return returns

    def compute_total_return(self):
        """
        Méthode permettant de calculer le total return
        """
        tot_ret: pd.DataFrame = (1 + self.returns).cumprod() - 1
        return tot_ret

    def compute_annualized_return(self):
        """
        Méthode permettant de calculer le rendement annualisé d'une stratégie.
        Intérêt dans le cadre du projet : comparer la performance du nouveau pilier (action / obligation) avec
        celui de la grille de départ
        """
        ann_ret: float = (self.nav[-1] / self.nav[0]) ** (self.annualization_factor / self.nav.shape[0]) - 1
        return ann_ret

    def compute_annualized_vol(self):
        """
        Méthode permettant de calculer la volatilité annualisée d'une stratégie
        """
        vol: float = np.std(self.returns)
        return vol * np.sqrt(self.annualization_factor)

    def compute_sharpe_ratio(self, rf:float = 0.02):
        """
        Méthode permettant de calculer le sharpe ratio de l'investissement
        """
        ann_ret:float = self.compute_annualized_return()
        ann_vol: float = self.compute_annualized_vol()
        sharpe:float = (ann_ret - rf)/ann_vol
        return sharpe

    """
    Méthode pour calculer la CVaR et Gini à mettre en place
    Peut être une méthode pour calculer les corrélations
    
    Rolling Window ou Expanding window ? Dans un cas, gestion de la taille
    """
    def compute_CVAR(self):
        """
        Méthode permettant de calculer la CVaR du portefeuille
        """