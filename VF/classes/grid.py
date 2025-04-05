import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
from typing import Tuple, List, Dict, Optional, Union, Any
from concurrent.futures import ThreadPoolExecutor

class Grid:
    """
    Classe représentant la grille de désensibilisation. Cette classe implémente : 
        - le calcul de statistiques descriptives et de CVaR par horizon de la grille, 
        - l'optimisation des rendements de la grille sous contrainte de CVaR cible. 

    Attributs à renseigner :
    ------------
        - periodicity (str) : périodicité à utiliser pour les calculs, 
        - long_term_returns (list) : liste des rendements de long terme hypothétiques pour chacun des 3 piliers,
        - portfolio_returns (pd.DataFrame) : rendements réels des piliers,
        - grid_weights (pd.DataFrame) : matrice des poids des piliers de la grille par horizon 
    """
    ANNUALISATION_FACTORS = {
        "yearly": 1,
        "quarterly": 4,
        "monthly": 12,
        "daily": 252
    }

    def __init__(self, long_term_returns: list, portfolios_returns: pd.DataFrame, periodicity:str = "yearly", grid_weights:pd.DataFrame = None):
        if periodicity not in Grid.ANNUALISATION_FACTORS.keys():
            raise ValueError("Problème de périodicité !")
        
        self.periodicity: str = periodicity
        self.grid_weights: pd.DataFrame = grid_weights.copy() if grid_weights is not None else self.optimize_grid()

        self.annualisation_factor: int = Grid.ANNUALISATION_FACTORS[periodicity]
        self.long_term_returns: list = long_term_returns.copy()
        self.portfolios_returns: pd.DataFrame = portfolios_returns.copy()

        # Variance, corrélation et covariance des rendements des 3 piliers
        self.portfolios_variance = self.portfolios_returns.var() * self.annualisation_factor
        self.portfolios_volatility = self.portfolios_returns.std() * np.sqrt(self.annualisation_factor)
        self.portfolios_correlations = self.portfolios_returns.corr()
        self.portfolios_returns_covariance = self.portfolios_returns.cov() * self.annualisation_factor

        # Rendement, variance et CVaR de la grille à chaque date
        self.grid_returns = (self.grid_weights * self.long_term_returns).sum(axis = 1)
        self.grid_variance = np.zeros(self.grid_weights.shape[0])
        self.grid_volatility = np.zeros(self.grid_weights.shape[0])
        self.grid_CVaR = pd.DataFrame(index=self.grid_weights.index, columns=['CVaR'])
        
        for i, weights in enumerate(self.grid_weights.values):
            self.grid_variance[i] = weights.T @ self.portfolios_returns_covariance @ weights
            self.grid_volatility[i] = np.sqrt(self.grid_variance[i])
            self.grid_CVaR.iloc[i,0] = self.compute_parametric_CVaR(self.grid_returns[i], self.grid_volatility[i])

    def compute_parametric_CVaR(self, grid_expected_return: float, grid_volatility: float, confidence_level:float = 0.05):
        """
        Fonction qui permet le calcul de la CVaR paramétrique en %. 
        La CVaR est calculée sur la base du rendement, de la volatilité et du seuil donnés.

        Inputs : 
        -------------
            - grid_expected_returns (float) : rendement espéré à l'horizon considéré,
            - grid_volatility (float) : volatilité à l'horizon considéré, 
            - confidence_level (float) : probabilité de la queue de distribution que l'on souhaite (5% par défaut)
        
        Output : 
        -------------
            - cvar (float) : Conditionnal Value at Risk
        """
        
        z_score = norm.ppf(confidence_level)
        phi_z = norm.pdf(z_score)
        cvar = ((1/confidence_level) * phi_z * grid_volatility) - grid_expected_return
        return cvar 

    def optimize_grid(self, initial_grid: 'Grid') -> 'Grid':
        """
        Optimise la grille de désensibilisation en maximisant les rendements sous contrainte de CVaR. L'optimisation se fait de manière glissante 
        pour chaque horizon considéré. 

        Input : 
        --------------
            - initial_grid (Grid) : grille de référence, qui définit la CVaR à respecter. 
        
        Output : 
        --------------
            optimized_grid (Grid) : instance de "Grid" avec les poids optimisés.

        """
        # Initialisation : nombre de piliers, d'horizon et CVaR cible
        n_ptf: int = self.grid_weights.shape[1]
        horizons = initial_grid.grid_weights.index
        target_CVaRs = initial_grid.grid_CVaR
    
        optimized_weights = pd.DataFrame(
            index=horizons,
            columns=self.grid_weights.columns,
            dtype=float
            )
    
        optimized_stats = pd.DataFrame(
            index=horizons,
            columns=['Rendement', 'Volatilité', 'CVaR'],
            dtype=float
            )

        # Contrainte 1 : pas de vente à découvert
        bounds = [(0, 1) for _ in range(n_ptf)]

        def optimize_single_horizon(horizon: int) -> Tuple[Union[int, str], np.ndarray, float, float, float, bool]:
            """
            Fonction qui optimise les poids pour un horizon spécifique. Elle sera donc appelée pour chaque horizon.

            Input : 
            ------------
                - horizon (int) : horizon temporel à optimiser
            """
            target_cvar = target_CVaRs.loc[horizon, 'CVaR']
            initial_weights = self.grid_weights.loc[horizon].values

            def objective_function(weights: np.ndarray) -> float:
                """
                Fonction objectif : maximisation des rendements de la grille.
                On utilise son opposée car scipy.stats minimise. 
                """
                return -np.dot(weights, self.long_term_returns)
        
            def cvar_constraint(weights: np.ndarray) -> float:
                """
                Fonction qui permet la création dynamique de contrainte de CVaR
                """
                ptf_expected_return = np.dot(weights, self.long_term_returns)
                # Calcul de la volatilité du portfolio avec ces poids
                ptf_volatility = np.sqrt(weights @ self.portfolios_returns_covariance @ weights)
                # Calcul de la CVaR 
                calculated_cvar = self.compute_parametric_CVaR(ptf_expected_return, ptf_volatility)
                # La contrainte est satisfaite si la CVaR calculée est inférieure ou égale à la CVaR cible
                return target_cvar - calculated_cvar
        
            # Contraintes 2 et 3 : somme des poids égale à 1 et CVaR au plus égale à celle de l'ancienne grille.
            constraints = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1},  
                {"type": "ineq", "fun": cvar_constraint}
            ]

            # Minimisation en utilisant les poids de la grille initiale comme point de départ
            result = minimize(
                objective_function,
                initial_weights,  
                method="SLSQP",
                bounds=bounds,
                constraints=constraints
            )

            if result.success:
                new_weights = result.x
                new_returns = np.dot(new_weights, self.long_term_returns)
                new_vol = np.sqrt(new_weights @ self.portfolios_returns_covariance @ new_weights)
                new_cvar = self.compute_parametric_CVaR(new_returns, new_vol)
                return horizon, new_weights, new_returns, new_vol, new_cvar, True

            # Si l'optimisation échoue, on l'indique
            else:
                print(f"L'optimisation a échoué pour l'horizon {horizon}. Message: {result.message}")
                # On utilise les poids initiaux en cas d'échec
                fake_weights = initial_weights
                ret = np.dot(fake_weights, self.long_term_returns)
                vol = np.sqrt(fake_weights @ self.portfolios_returns_covariance @ fake_weights)
                cvar = self.compute_parametric_CVaR(ret, vol)
                return horizon, fake_weights, ret, vol, cvar, False

        # Parallélisation des optimisations
        with ThreadPoolExecutor(max_workers=min(10, len(horizons))) as executor:
            results = list(executor.map(optimize_single_horizon, horizons))

        # Renvoi des résultats
        for horizon, weights, ret, vol, cvar, success in results:
            optimized_weights.loc[horizon] = weights
            optimized_stats.loc[horizon, 'Rendement'] = ret
            optimized_stats.loc[horizon, 'Volatilité'] = vol
            optimized_stats.loc[horizon, 'CVaR'] = cvar

        optimized_grid = Grid(self.long_term_returns, self.portfolios_returns, self.periodicity, optimized_weights)
        successful = sum(1 for _, _, _, _, _, success in results if success)
        print(f"Optimisation réussie pour {successful} horizons sur {len(horizons)}")
        return optimized_grid

