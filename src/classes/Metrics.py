import numpy as np
import pandas as pd

class Metrics():

    DISCRET_LABEL = "discret"
    CONTINU_LABEL = "continu"
    DAILY_LABEL = "daily"
    WEEKLY_LABEL = "weekly"
    MONTHLY_LABEL = "monthly"
    QUARTERLY_LABEL = "quarterly"
    YEARLY_LABEL = "yearly"

    """
    Classe permettant de calculer un ensemble de métriques de performance et de risque pour un portefeuille
    Input : 
        - ptf_nav, une liste contenant l'évolution de la NAV du portefeuille dans le temps
        - method : méthode pour calculer les rendements
        - frequency : fréquence à utiliser pour le calcul des métriques
        - benchmark : dataframe contenant le benchmark si on souhaite utiliser cette classe
        pour faire des comparaisons portefeuille / benchmark
    """
    def __init__(self, ptf_nav:pd.DataFrame, method:str, frequency:str, benchmark:pd.DataFrame = None):
        self.nav: pd.DataFrame = ptf_nav
        self.bench: pd.DataFrame = benchmark
        self.method: str = method
        self.frequency: str = frequency
        self.annualization_factor: int = self.compute_annualization_factor()

    def compute_annualization_factor(self):
        """
        Méthode permettant de déterminer le coefficient d'annualisation
        à utiliser selon la périodicité des données
        """
        if self.frequency == Metrics.DAILY_LABEL:
            return 252
        elif self.frequency == Metrics.WEEKLY_LABEL:
            return 52
        elif self.frequency == Metrics.MONTHLY_LABEL:
            return 12
        elif self.frequency == Metrics.QUARTERLY_LABEL:
            return 4
        elif self.frequency == Metrics.YEARLY_LABEL:
            return 1
        else:
            raise Exception("Les calculs pour une périodicité autre ne sont pas implémentés")

    def compute_returns(self, nav: pd.DataFrame):
        """
        Méthode permettant de calculer les rendements du portefeuille
        selon la méthode spécifiée
        """

        period_resampling: dict = {
            "daily": None,
            "weekly": 'W-FRI',
            "monthly": 'ME',
            "quarterly": 'QE',
            "yearly": 'YE'
        }
        resampled_data: pd.DataFrame = nav if period_resampling[self.frequency] is None else nav.resample(
            period_resampling[self.frequency]).last()
        returns: pd.DataFrame = resampled_data.pct_change().dropna() if self.method == "discret" else np.log(
            resampled_data).diff().dropna()
        return returns

    def compute_performance(self, nav: pd.DataFrame, ret:pd.DataFrame):
        """
        Méthode permettant de calculer le rendement annualisé d'une stratégie.
        Intérêt dans le cadre du projet : comparer la performance du nouveau pilier (action / obligation) avec
        celui de la grille de départ
        """
        # Calcul du total return
        total_return: float = (nav.iloc[-1] / nav.iloc[0]) - 1

        # Calcul du rendement annualisé
        annualized_return: float = (1+total_return) ** (self.annualization_factor / ret.shape[0]) - 1
        return {"total_return": total_return, "annualized_return": annualized_return}

    def compute_annualized_vol(self, ret:pd.DataFrame):
        """
        Méthode permettant de calculer la volatilité annualisée d'une stratégie
        """
        vol: float = np.std(ret)
        return vol * np.sqrt(self.annualization_factor)

    def compute_sharpe_ratio(self, nav: pd.DataFrame, ret:pd.DataFrame, rf:float = 0.02):
        """
        Méthode permettant de calculer le sharpe ratio de l'investissement
        """
        ann_ret:float = self.compute_performance(nav, ret)["annualized_return"]
        ann_vol: float = self.compute_annualized_vol(ret)
        sharpe:float = (ann_ret - rf)/ann_vol
        return sharpe

    def compute_premia(self):
        """
        Calcul de la prime de diversification en terme de sharpe ratio
        ==> voir pour faire évoluer la méthode
        """
        bench_ret: pd.DataFrame = self.compute_returns(self.bench)
        ptf_ret: pd.DataFrame = self.compute_returns(self.nav)

        # Calcul du sharpe des deux portefeuilles
        ptf_sharpe: float = self.compute_sharpe_ratio(self.nav, ptf_ret)
        bench_sharpe: float = self.compute_sharpe_ratio(self.bench, bench_ret)

        # Calcul de la prime
        premia:float = ptf_sharpe - bench_sharpe
        print(f"La prime de diversification s'élève à {round(premia * 100, 2)} % pour chaque point de volatilité supplémentaire pris")
        return premia

    def synthesis(self):
        """
        Pour une comparaison portefeuille bench : synthèse des éléments et calcul
        de la prime de diversification
        """
        print("Statistiques pour le portefeuille : ")
        self.display_stats()
        print("Statistiques pour le benchmark  :")
        self.display_bench_stats()
        premia: float = self.compute_premia()
        return premia

    def display_stats(self):
        """
        Méthode permettant d'afficher les différentes statistiques descriptives du portefeuille
        """
        ptf_ret = self.compute_returns(self.nav)
        ann_ret: float = self.compute_performance(self.nav,ptf_ret)["annualized_return"]
        ann_vol: float = self.compute_annualized_vol(ptf_ret)
        tot_ret: float = self.compute_performance(self.nav, ptf_ret)["total_return"]
        sharpe: float = self.compute_sharpe_ratio(self.nav, ptf_ret)

        print(f"Rendement annualisé du portefeuille (en %) : {round(ann_ret * 100, 2)}.")
        print(f"Volatilité annualisée du portefeuille (en %) : {round(ann_vol * 100, 2)}.")
        print(f"Sharpe ratio du portefeuille : {round(sharpe,2)}.")
        print(f"Total return du portefeuille (en %) : {round(tot_ret * 100, 2)}")

    def display_bench_stats(self):

        """
        Méthode pour afficher les statistiques du benchmark
        """
        if self.bench.empty:
            raise Exception("Aucun benchmark renseigné")

        bench_ret: pd.DataFrame = self.compute_returns(self.bench)
        ann_ret: float = self.compute_performance(self.bench, bench_ret)["annualized_return"]
        ann_vol: float = self.compute_annualized_vol(bench_ret)
        tot_ret: float = self.compute_performance(self.bench, bench_ret)["total_return"]
        sharpe: float = self.compute_sharpe_ratio(self.bench, bench_ret)

        print(f"Rendement annualisé du benchmark (en %) : {round(ann_ret * 100, 2)}.")
        print(f"Volatilité annualisée du benchmark (en %) : {round(ann_vol * 100, 2)}.")
        print(f"Sharpe ratio du benchmark : {round(sharpe,2)}.")
        print(f"Total return du benchmark (en %) : {round(tot_ret * 100, 2)}")

