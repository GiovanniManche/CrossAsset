from datetime import datetime
import pandas as pd
import math
from abc import ABC, abstractmethod

class Strategy(ABC):
    def generate_signal(self, data_for_signal_generation:dict):
        """
        Méthode abstraite permettant de générer des signaux d'investissement selon une stratégie donnée.
        Input : data_for_signal_generation : un dictionnaire qui utilise des tickers comme clé et les positions comme valeurs
        Output : dictionnaire contenant 
        """
        pass