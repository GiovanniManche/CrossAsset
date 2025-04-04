import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
from typing import Tuple, List, Dict, Optional, Union, Any
from concurrent.futures import ThreadPoolExecutor

class Grid:
    ANNUALISATION_FACTORS = {
        "yearly": 1,
        "quarterly": 4,
        "monthly": 12,
        "daily": 252
    }

    def __init__(self, long_term_returns: list, portfolios_returns: pd.DataFrame, periodicity:str = "yearly", grid_weights:pd.DataFrame = None):
        if periodicity not in Grid.ANNUALISATION_FACTORS.keys():
            raise ValueError("Problème de périodicité !")
        
        self.periodicity = periodicity
        self.grid_weights = grid_weights.copy() if grid_weights is not None else self.optimize_grid()

        self.annualisation_factor = Grid.ANNUALISATION_FACTORS[periodicity]
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

    def compute_parametric_CVaR(self, grid_expected_return: float, grid_volatility: float, confidence_level:float = 0.05, initial_investment: float = 100.0):
        z_score = norm.ppf(confidence_level)
        phi_z = norm.pdf(z_score)
        cvar = ((1/confidence_level) * phi_z * grid_volatility) - grid_expected_return
        return cvar 

    def optimize_grid(self, initial_grid: 'Grid') -> 'Grid':
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

        bounds = [(0, 1) for _ in range(n_ptf)]

        def optimize_single_horizon(horizon: Union[int, str]) -> Tuple[Union[int, str], np.ndarray, float, float, float, bool]:
            target_cvar = target_CVaRs.loc[horizon, 'CVaR']
            initial_weights = self.grid_weights.loc[horizon].values

            def objective_function(weights: np.ndarray) -> float:
                return -np.dot(weights, self.long_term_returns)
        
            def cvar_constraint(weights: np.ndarray) -> float:
                ptf_expected_return = np.dot(weights, self.long_term_returns)
            
            # Calcul de la volatilité du portfolio avec ces poids
                ptf_volatility = np.sqrt(weights @ self.portfolios_returns_covariance @ weights)
            
            # Calcul de la CVaR avec la fonction existante
                calculated_cvar = self.compute_parametric_CVaR(ptf_expected_return, ptf_volatility)
            
            # La contrainte est satisfaite si la CVaR calculée est inférieure ou égale à la CVaR cible
                return target_cvar - calculated_cvar
        
            constraints = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1},  # somme des poids = 1
                # CVaR de la nouvelle grille inférieure ou égale à celle de l'ancienne
                {"type": "ineq", "fun": cvar_constraint}
            ]

            result = minimize(
                objective_function,
                initial_weights,  # Utiliser les poids initiaux comme point de départ
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
        
            else:
                print(f"L'optimisation a échoué pour l'horizon {horizon}. Message: {result.message}")
                # Utiliser les poids initiaux en cas d'échec
                fake_weights = initial_weights
                ret = np.dot(fake_weights, self.long_term_returns)
                vol = np.sqrt(fake_weights @ self.portfolios_returns_covariance @ fake_weights)
                cvar = self.compute_parametric_CVaR(ret, vol)
                return horizon, fake_weights, ret, vol, cvar, False

        # Utiliser un ThreadPoolExecutor pour paralléliser les optimisations
        with ThreadPoolExecutor(max_workers=min(10, len(horizons))) as executor:
            results = list(executor.map(optimize_single_horizon, horizons))

        # Traiter les résultats
        for horizon, weights, ret, vol, cvar, success in results:
            optimized_weights.loc[horizon] = weights
            optimized_stats.loc[horizon, 'Rendement'] = ret
            optimized_stats.loc[horizon, 'Volatilité'] = vol
            optimized_stats.loc[horizon, 'CVaR'] = cvar

    # Création de la nouvelle grille optimisée
        optimized_grid = Grid(self.long_term_returns, self.portfolios_returns, self.periodicity, optimized_weights)

    # Résumer les résultats
        successful = sum(1 for _, _, _, _, _, success in results if success)
        print(f"Optimisation réussie pour {successful} horizons sur {len(horizons)}")
    
        return optimized_grid

