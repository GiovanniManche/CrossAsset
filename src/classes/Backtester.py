import pandas as pd
import datetime as datetime
from dateutil.relativedelta import relativedelta


class Backtester:

    DAILY_LABEL = "Daily"
    WEEKLY_LABEL = "Weekly"
    MONTHLY_LABEL = "Monthly"
    QUARTERLY_LABEL = "Quarterly"
    YEARLY_LABEL = "Yearly"

    """
    Classe permettant d'effectuer un backtest pour construire un portefeuille action ou crédit.
    Utilité Construire les portefeuilles pour chaque pilier (action / obligation)
    """

    def __init__(self, rebal:str, method:str, file_path:str):
        self.rebalancement: str = rebal
        self.method: str = method
        self.prices:pd.DataFrame = self._load_data(file_path)
        self.start_date:datetime = self.prices.index(0)

        # Création des dictionnaires
        self.portfolio: dict = dict()
        self.portfolio_long: dict = dict()
        self.portfolio_short: dict = dict()
        self.ptf_weights: dict = dict()
        self.ptf_long_weights: dict = dict()
        self.ptf_short_weights:dict = dict()

        # Benchmark = indice de référence donné dans la grille initiale

    @staticmethod
    def _load_data(file_path:str):
        """
        Méthode permettant de sélectionner les données à utilisée
        """

        # Version simple avec un seul fichier, à compléter éventuellement ultérieurement
        try:
            df_prices: pd.DataFrame = pd.read_excel(file_path, sheet_name="Valeus indices clean", skiprows=2)
        except Exception:
            raise Exception("Erreur dans l'import des données du fichier excel")
        return df_prices

    def rebalancing_date(self, prec_date:datetime = None):
        """
        Méthode permettant de calculer la nouvelle date de rebalancement
        à partir de la date de rebalancement précédente
        """

        # On rebalance à la première itération du backtest (=> à voir, peut équipondérer sinon)
        if prec_date == None:
            return self.start_date

        else:
            if self.rebalancement == Backtester.DAILY_LABEL:
                next_rebalancing_date:datetime = prec_date.timedelta(days = 1)
            elif self.rebalancement == Backtester.WEEKLY_LABEL:
                next_rebalancing_date:datetime = prec_date.timedelta(days = 7)
            elif self.rebalancement == Backtester.MONTHLY_LABEL:
                next_rebalancing_date:datetime = prec_date + relativedelta(months=1)
            elif self.rebalancement == Backtester.QUARTERLY_LABEL:
                next_rebalancing_date:datetime = prec_date + relativedelta(months = 3)
            elif self.rebalancement == Backtester.YEARLY_LABEL:
                next_rebalancing_date:datetime = prec_date + relativedelta(years = 1)
            else:
                raise Exception("La fréquence de rebalancement souhaitée n'est pas implémentée")

        return next_rebalancing_date

    def compute_nav(self):
        """
        Méthode permettant de calculer la NAV du portefeuille
        """



    def run_backtest(self):
        """
        Méthode permettant de lancer
        le backtest pour construire le portefeuille
        """


