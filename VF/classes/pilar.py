import pandas as pd
import numpy as np
from scipy.optimize import minimize
from datetime import timedelta

from classes.portfolio import Portfolio

class Pilar:
    """
    Classe qui correspond à un pilier de la grille (Actions, Obligations, Monétaire).
    Inputs : 
    --------------
        - asset_prices : prix des actifs qui composent le pilier. Par exemple, pour le pilier actions,
        nous mettons les prix des actifs core et du portefeuille Momentum créé. 
        - periodoicity : indique la périodicité des rendements pour les calculs
        - method : rendements discrets VS rendements continus
    
    Fonctionnalités : 
    --------------
        - compute_asset_returns : calcule les rendements des actifs sur la base de la méthode et périodicitié
        indiquées lors de l'instanciation
        - rolling_mean_variance_optimization : optimise le portefeuille en visant exclusivement à réduire la variance.
        On raisonne sur fenêtre roulante.
        - compute_pilar_returns : calcule les rendements du pilier à partir des poids calculés ou donnés.
        - run_min_variance_optimization : lance la minimisation roulante de la variance sous contrainte, et renvoie les poids obtenus. 
        - compute_pilar_value : calcule la valeur du pilier ainsi obtenu
    """
    METHODS_LABELS = ["discret","continu"]

    DAILY_LABEL = "daily"
    WEEKLY_LABEL = "weekly"
    MONTHLY_LABEL = "monthly"
    QUARTERLY_LABEL = "quarterly"
    YEARLY_LABEL = "yearly"

    def __init__(self, asset_prices: pd.DataFrame, periodicity: str, method: str) -> None:
        # Vérifications de cohérence
        if periodicity.lower() not in Portfolio.PERIODICITY_LABELS:
            raise Exception(f"La périodicité pour le calcul des rendements doit être l'une des suivantes : {Portfolio.PERIODICITY_LABELS}.")
        if method.lower() not in Portfolio.METHODS_LABELS:
            raise Exception(f"La méthode pour le calcul des rendements doit être l'une des suivantes : {Portfolio.METHODS_LABELS}.")
        self.asset_prices: pd.DataFrame = asset_prices
        self.method: str = method
        self.returns: pd.DataFrame = self.compute_asset_returns(periodicity)

    def compute_asset_returns(self, periodicity_returns:str) -> pd.DataFrame:
        """
        Calcule les rendements des actifs de l'univers en fonction de la périodicité et de la méthode
        
        Input : 
        --------------
            - str: periodicity_returns : périodicité à prendre pour le calcul des rendements.
        
        Output : 
        --------------
            - pd.DataFrame: returns : rendements des actifs du pilier selon la méthode et périodicité demandées
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
        
    def rolling_min_variance_optimization(self, calculation_window: int, 
                                          rebalancing_freq: str, min_weights: dict = None) -> pd.DataFrame:
        """
        Optimise un portefeuille en minimisant la variance sous contrainte. 
        Les statistiques nécessaires sont calculées sur une fenêtre glissante.
        Retourne les poids pour toutes les dates, pas seulement aux dates de rebalancement.
        
        Inputs :
        --------------
        calculation_window : int
            Taille de la fenêtre glissante en nombre de périodes
        rebalancing_freq : str
            Fréquence de rebalancement ('quarterly', 'monthly', etc.)
        min_weights : dict
            Dictionnaire spécifiant les contraintes de poids minimaux pour certains actifs
            Format: {'SPX Index': 0.2, 
                    'NKY Index': 0.3,...}
            
        Outputs :
        --------------
        weights : pd.DataFrame
            DataFrame contenant les poids optimaux pour toutes les dates dans l'index des rendements
        """
        # Gestion du rebalancement
        if rebalancing_freq.lower() == self.QUARTERLY_LABEL:
            rebalancing_df: pd.DataFrame = self.returns.resample('QE').last()
            rebalancing_dates: list = rebalancing_df.index.intersection(self.returns.index)
        elif rebalancing_freq.lower() == self.MONTHLY_LABEL:
            rebalancing_df: pd.DataFrame = self.returns.resample('ME').last()
            rebalancing_dates: list = rebalancing_df.index.intersection(self.returns.index)
        else:
            raise ValueError(f"La fréquence de rebalancement {rebalancing_freq} n'est pas supportée.")
            
        # Initialisation du DataFrame pour stocker les poids optimaux aux dates de rebalancement
        assets: list = self.returns.columns
        rebal_weights: pd.DataFrame = pd.DataFrame(index=rebalancing_dates, columns=assets)
        
        # On cherche à minimiser la variance du portefeuille
        def objective_function(weights: pd.DataFrame, cov_matrix: pd.DataFrame) -> float:
            portfolio_variance: float = np.dot(weights.T, np.dot(cov_matrix, weights))
            return portfolio_variance
            
        # Première contrainte : somme des poids = 1
        def sum_constraint(weights: pd.DataFrame) -> float:
            return np.sum(weights) - 1.0
            
        # On change les poids à chaque date de rebalancement
        for date in rebalancing_dates:
            # On vérifie qu'on a assez de données
            loc_in_index: int = self.returns.index.get_loc(date)
            if loc_in_index >= calculation_window - 1: 
                # Sélection des données dans la fenêtre
                end_idx: int = loc_in_index
                start_idx: int = end_idx - calculation_window + 1
                window_slice: list = slice(self.returns.index[start_idx], self.returns.index[end_idx])
                
                returns_window: pd.DataFrame = self.returns.loc[window_slice]
                
                # Matrice de covariance
                cov_matrix: pd.DataFrame = returns_window.cov()
                
                # Vérification que la matrice de covariance est définie positive
                # Si non, on ajoute une petite valeur à la diagonale
                min_eig = np.min(np.linalg.eigvals(cov_matrix))
                if min_eig <= 0:
                    # Ajouter une petite valeur à la diagonale pour rendre la matrice définie positive
                    cov_matrix = cov_matrix + np.eye(len(assets)) * abs(min_eig) * 1.1
                
                # Nombre d'actifs
                n_assets = len(assets)
                
                # Définition de la contrainte sur la somme des poids = 1
                constraints = [
                    {'type': 'eq', 'fun': sum_constraint}  
                ]
                # Création des bornes avec les contraintes de poids minimaux
                bounds = []
                for asset in assets:
                    # Si min_weights existe, on récupère le poids minimal donné, sinon on le fixe à 0
                    min_weight = min_weights.get(asset, 0.0) if min_weights else 0.0
                    bounds.append((min_weight, 1.0))
                
                # On vérifie que la somme des poids minimaux ne dépasse pas 1
                sum_min_weights = sum(min_weight for asset, 
                                      min_weight in min_weights.items() if asset in assets
                                      ) if min_weights else 0
                if sum_min_weights > 1:
                    raise ValueError(f"Erreur à la date {date}: La somme des poids minimaux ({sum_min_weights}) dépasse 1.")
                    
                
                # Initialisation : allocation équipondérée
                initial_weights = np.ones(n_assets) / n_assets
                
                try:
                    # Optimisation
                    result = minimize(
                        objective_function,
                        initial_weights,
                        args=(cov_matrix,),
                        method='SLSQP',
                        bounds=bounds,
                        constraints=constraints,
                        options={'maxiter': 1000, 'disp': False, 'ftol': 1e-8}
                    )
                    
                    # Stockage des poids optimaux
                    if result['success']:
                        # Nettoyage des poids très légèrement négatifs (erreurs numériques)
                        weights = np.maximum(result['x'], 0)
                        # On normalise pour s'assurer que la somme est exactement 1
                        weights = weights / np.sum(weights)
                        # Ajout des poids 
                        rebal_weights.loc[date] = weights
                    else:
                        print(f"L'optimisation a échoué à la date {date}: {result['message']}")
                        # On essaye une autre méthode
                        try:
                            result = minimize(
                                objective_function,
                                initial_weights,
                                args=(cov_matrix,),
                                method='trust-constr',
                                bounds=bounds,
                                constraints=constraints,
                                options={'maxiter': 10000, 'disp': False}
                            )
                            
                            if result['success']:
                                # Nettoyage des poids
                                weights = np.maximum(result['x'], 0)
                                weights = weights / np.sum(weights)
                                rebal_weights.loc[date] = weights
                            else:
                                # Si rien ne fonctionne, on utilise une allocation initiale équipondérée
                                rebal_weights.loc[date] = initial_weights
                        except:
                            rebal_weights.loc[date] = initial_weights
                except Exception as e:
                    print(f"Erreur lors de l'optimisation à la date {date}: {str(e)}")
                    # Equipondération en cas d'erreur
                    rebal_weights.loc[date] = initial_weights
        
        # Nettoyage des entrées NaN au début
        rebal_weights = rebal_weights.dropna()
        
        if len(rebal_weights) == 0:
            raise ValueError("Aucune date de rebalancement valide n'a été trouvée. Vérifiez vos données et paramètres.")
        
        # Maintenant, on doit propager les poids à toutes les dates dans l'index des rendements en tenant compte de la dérive
        # Initialisation du DataFrame pour toutes les dates
        all_weights = pd.DataFrame(index=self.returns.index, columns=assets)
        sorted_rebal_dates = sorted(rebal_weights.index)
        
        # Traitement entre les dates de rebalancement
        for i in range(len(sorted_rebal_dates) - 1):
            start_date = sorted_rebal_dates[i]
            end_date = sorted_rebal_dates[i+1]
            
            # Sélection des dates entre deux rebalancements 
            mask = (self.returns.index >= start_date) & (self.returns.index < end_date)
            dates_between = self.returns.index[mask]
            current_weights = rebal_weights.loc[start_date].values
            
            # On assigne les poids initiaux à la date de rebalancement
            all_weights.loc[start_date] = current_weights
            
            # Pour chaque date entre les rebalancements, on calcule la dérive
            prev_date = start_date
            for current_date in dates_between[1:]:  
                # On récupère les rendements pour la période entre prev_date et current_date
                if prev_date in self.returns.index and current_date in self.returns.index:
                    prev_idx = self.returns.index.get_loc(prev_date)
                    current_idx = self.returns.index.get_loc(current_date)
                    
                    if current_idx > prev_idx:
                        # On récupère les rendements pour cette période
                        period_returns = self.returns.iloc[prev_idx]
                        
                        # On calcule les nouveaux poids après dérive
                        # Etape 1: calcul du rendement du portefeuille
                        portfolio_return = np.sum(current_weights * period_returns)
                        total_return = 1 + portfolio_return
                        
                        # Etape 2: on calcule les nouveaux poids après dérive
                        new_weights = []
                        for j, asset_return in enumerate(period_returns):
                            old_weight = current_weights[j]
                            total_return_asset = 1 + asset_return
                            new_weight = old_weight * total_return_asset / total_return
                            new_weights.append(new_weight)
                        
                        # MAJ
                        current_weights = np.array(new_weights)
                        all_weights.loc[current_date] = current_weights
    
                prev_date = current_date
        
        # Traitement après le dernier rebalancement
        if len(sorted_rebal_dates) > 0:
            last_rebal_date = sorted_rebal_dates[-1]
            
            all_weights.loc[last_rebal_date] = rebal_weights.loc[last_rebal_date].values
            
            # On calcule la dérive pour les dates après le dernier rebalancement
            mask = self.returns.index > last_rebal_date
            post_dates = self.returns.index[mask]
            
            # Poids au dernier rebalancement
            current_weights = rebal_weights.loc[last_rebal_date].values
            prev_date = last_rebal_date
            for current_date in post_dates:
                # Récupération des rendements
                if prev_date in self.returns.index and current_date in self.returns.index:
                    prev_idx = self.returns.index.get_loc(prev_date)
                    current_idx = self.returns.index.get_loc(current_date)
                    
                    if current_idx > prev_idx:
                        period_returns = self.returns.iloc[prev_idx]
                        
                        # Calcul des nouveaux poids après dérive (cf portfolio pour méthodologie)
                        portfolio_return = np.sum(current_weights * period_returns)
                        total_return = 1 + portfolio_return
                        
                        new_weights = []
                        for j, asset_return in enumerate(period_returns):
                            old_weight = current_weights[j]
                            total_return_asset = 1 + asset_return
                            new_weight = old_weight * total_return_asset / total_return
                            new_weights.append(new_weight)
                        
                        # Mise à jour des poids
                        current_weights = np.array(new_weights)
                        all_weights.loc[current_date] = current_weights
    
                prev_date = current_date
        
        # Suppression des lignes avec des NaN (dates avant le premier rebalancement)
        all_weights = all_weights.dropna()
        
        return all_weights
    
    def compute_pilar_returns(self, weights: pd.DataFrame) -> pd.Series:
        """
        Calcule les rendements du pilier en utilisant les poids disponibles pour chaque date.
        
        Inputs :
        -----------
        weights : pd.DataFrame
            DataFrame contenant les poids optimaux pour chaque date de calcul
            
        Outputs :
        --------
        returns : pd.Series
            Série des rendements du portefeuille
        """
        # Vérifie que weights n'est pas vide
        if weights.empty:
            raise ValueError("Le DataFrame des poids est vide")
            
        # Initialisation de la série des rendements du portefeuille
        portfolio_returns = pd.Series(index=weights.index, dtype=float)
        
        # Calcul des rendements du portefeuille pour chaque date
        for date in weights.index:
            if date in self.returns.index:  
                portfolio_returns.loc[date] = np.sum(self.returns.loc[date] * weights.loc[date])
        
        return portfolio_returns.dropna()
   
    def run_min_variance_optimization(self, calculation_window: int, rebalancing_freq: str, min_weights: dict = None):
        """
        Exécute l'ensemble du processus d'optimisation à variance minimale et de backtest
        
        Parameters:
        -----------
        calculation_window: int
            Taille de la fenêtre glissante en nombre de périodes
        rebalancing_freq: str
            Fréquence de rebalancement ('quarterly', 'monthly', etc.)
        min_weights: dict
            Dictionnaire spécifiant les contraintes de poids minimaux pour certains actifs
            Format: {'asset_name': min_weight, ...}
            
        Returns:
        --------
        tuple
            (poids optimaux pour toutes les dates, rendements du portefeuille)
        """
        # Optimisation à variance minimale avec propagation des poids à toutes les dates
        optimal_weights = self.rolling_min_variance_optimization(
            calculation_window=calculation_window,
            rebalancing_freq=rebalancing_freq,
            min_weights=min_weights
        )
        
        # Avec les poids pour toutes les dates, nous pouvons directement calculer les rendements
        portfolio_returns = self.compute_pilar_returns(optimal_weights)
        
        return optimal_weights, portfolio_returns
    
    def compute_pilar_value(self, weights: pd.DataFrame, initial_value: float = 100.0) -> pd.Series:
        """
        Calcule la valeur du pilier à partir des poids optimaux et d'une valeur initiale
        
        Parameters:
        -----------
        weights: pd.DataFrame
            DataFrame contenant les poids optimaux pour toutes les dates
        initial_value: float
            Valeur initiale du portefeuille
            
        Returns:
        --------
        pd.Series
            Série contenant la valeur du pilier à chaque date
        """
        # Vérifications élémentaires
        if weights.isna().any().any():
            raise ValueError("Les poids contiennent des NaN. Vérifiez les résultats de l'optimisation.")
            
        # On calcule les rendements du portefeuille
        portfolio_returns = self.compute_pilar_returns(weights)
        
        # On initialise la série de valeurs du portefeuille
        portfolio_value = pd.Series(index=portfolio_returns.index, dtype=float)
        
        # La première valeur est la valeur initiale
        portfolio_value.iloc[0] = initial_value
        
        # Pour chaque date 
        for i in range(1, len(portfolio_returns)):
            # Mise à jour de la valeur du portefeuille
            portfolio_value.iloc[i] = portfolio_value.iloc[i-1] * (1 + portfolio_returns.iloc[i])
        
        return portfolio_value
