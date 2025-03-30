import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
from typing import Tuple, List, Dict, Optional, Union, Any
from concurrent.futures import ThreadPoolExecutor
from Piliers import Pilier

class Grid:
    ANNUALISATION_FACTORS = {
        "yearly": 1,
        "quarterly": 4,
        "monthly": 12,
        "daily": 252
    }

    def __init__(self, long_term_returns: list, pilars_returns: pd.DataFrame, periodicity:str = "yearly", grid_weights:pd.DataFrame = None):
        if periodicity not in Grid.ANNUALISATION_FACTORS.keys():
            raise ValueError("Problème de périodicité !")
        
        self.periodicity = periodicity
        self.grid_weights = grid_weights.copy() if grid_weights is not None else self.optimize_grid()

        self.annualisation_factor = Grid.ANNUALISATION_FACTORS[periodicity]
        self.long_term_returns: list = long_term_returns.copy()
        self.pilars_returns: pd.DataFrame = pilars_returns.copy()

        # Variance, corrélation et covariance des rendements des 3 piliers
        self.pilars_returns_variance = self.pilars_returns.var() * self.annualisation_factor
        self.pilars_returns_volatility = self.pilars_returns.std() * np.sqrt(self.annualisation_factor)
        self.pilars_returns_correlation = self.pilars_returns.corr()
        self.pilars_returns_covariance = self.pilars_returns.cov() * self.annualisation_factor

        # Rendement, variance et CVaR du portefeuille à chaque date
        self.grid_returns = (self.grid_weights * self.long_term_returns).sum(axis = 1)
        self.grid_variance = np.zeros(self.grid_weights.shape[0])
        self.grid_volatility = np.zeros(self.grid_weights.shape[0])
        self.grid_CVaR = pd.DataFrame(index=self.grid_weights.index, columns=['CVaR'])
        
        for i, weights in enumerate(self.grid_weights.values):
            self.grid_variance[i] = weights.T @ self.pilars_returns_covariance @ weights
            self.grid_volatility[i] = np.sqrt(self.grid_variance[i])
            self.grid_CVaR.iloc[i,0] = self.compute_parametric_CVaR(self.grid_returns[i], self.grid_volatility[i])

    def compute_parametric_CVaR(self, ptf_expected_return: float, ptf_volatility: float, confidence_level:float = 0.05, initial_investment: float = 100.0):
        z_score = norm.ppf(confidence_level)
        phi_z = norm.pdf(z_score)
        cvar = ((1/confidence_level) * phi_z * ptf_volatility) - ptf_expected_return
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
                ptf_volatility = np.sqrt(weights @ self.pilars_returns_covariance @ weights)
            
            # Calcul de la CVaR avec la fonction existante
                calculated_cvar = self.compute_parametric_CVaR(ptf_expected_return, ptf_volatility)
            
            # La contrainte est satisfaite si la CVaR calculée est inférieure ou égale à la CVaR cible
                return target_cvar - calculated_cvar
        
            constraints = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1},  # somme des poids = 1
                # CVaR de la nouvelle grille au pire égale à celle de l'ancienne
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
                new_vol = np.sqrt(new_weights @ self.pilars_returns_covariance @ new_weights)
                new_cvar = self.compute_parametric_CVaR(new_returns, new_vol)
                return horizon, new_weights, new_returns, new_vol, new_cvar, True
        
            else:
                print(f"L'optimisation a échoué pour l'horizon {horizon}. Message: {result.message}")
                # Utiliser les poids initiaux en cas d'échec
                fake_weights = initial_weights
                ret = np.dot(fake_weights, self.long_term_returns)
                vol = np.sqrt(fake_weights @ self.pilars_returns_covariance @ fake_weights)
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
        optimized_grid = Grid(self.long_term_returns, self.pilars_returns, self.periodicity, optimized_weights)

    # Résumer les résultats
        successful = sum(1 for _, _, _, _, _, success in results if success)
        print(f"Optimisation réussie pour {successful} horizons sur {len(horizons)}")
    
        return optimized_grid

    def plot_weights_evolution(self, title: str = "Evolution de l'allocation en fonction du nombre d'années avant la retraite") -> plt.Axes:
        """
        Affiche l'évolution des poids du portefeuille sous forme de graphique à aires empilées.
        
        Args:
            title (str, optional): Titre du graphique. Défaut: "Evolution de l'allocation en fonction du nombre d'années avant la retraite".
            
        Returns:
            plt.Axes: L'objet Axes contenant le graphique.
        """
        plt.figure(figsize=(10, 4))

        ax = self.grid_weights.plot.area(figsize=(12, 6), alpha=0.7, stacked=True)

        # On inverse l'axe des abscisses, puis mettons titre, etc
        ax.invert_xaxis()
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Horizon avant la retraite (années)', fontsize=12)
        ax.set_ylabel('Allocation (%)', fontsize=12)
        ax.legend(title='Classes d\'actifs', loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_ylim(0, 1)  
        ax.grid(True, linestyle='--', alpha=0.7)

        # On ajoute une grille (tous les 20%)
        for y in np.arange(0.2, 1.0, 0.2):
            ax.axhline(y=y, color='gray', linestyle='-', alpha=0.3)

        # Si les poids sont en décimales (de 0 à 1), formatez l'axe y en pourcentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

        plt.tight_layout()
        return ax

   


"""
initial_grid = pd.read_excel("Inputs.xlsx", sheet_name="Grille initiale").set_index("Horizon")

prices_actions = pd.read_excel("Inputs.xlsx", sheet_name= "Actions").set_index("Date")
cols_with_na = [col for col in prices_actions.columns if prices_actions[col].isnull().sum() != 0]
prices_actions[cols_with_na] = prices_actions[cols_with_na].interpolate(method = 'linear')
pilier_actions = Pilier(prices_actions, "daily", "discret")

prices_obligation =  pd.read_excel("Inputs.xlsx", sheet_name= "Obligations").set_index("Date")
cols_with_na = [col for col in prices_obligation.columns if prices_obligation[col].isnull().sum() != 0]
prices_obligation[cols_with_na] = prices_obligation[cols_with_na].interpolate(method = 'linear')
pilier_obligations = Pilier(prices_obligation, "daily", "discret")

prices_monetaire =  pd.read_excel("Inputs.xlsx", sheet_name= "Monétaire").set_index("Date")
cols_with_na = [col for col in prices_monetaire.columns if prices_monetaire[col].isnull().sum() != 0]
prices_monetaire[cols_with_na] = prices_monetaire[cols_with_na].interpolate(method = 'linear')
pilier_monetaire = Pilier(prices_monetaire, "daily", "discret")

actions_ptf = pilier_actions.build_portfolio()
oblig_ptf = pilier_obligations.build_portfolio()
mon_ptf = pilier_monetaire.build_portfolio()
valeur_ptf_action = pilier_actions.compute_portfolio_value(actions_ptf, 100)
valeur_ptf_oblig = pilier_obligations.compute_portfolio_value(oblig_ptf, 100)
valeur_ptf_mon = pilier_monetaire.compute_portfolio_value(mon_ptf, 100)
returns_ptf_action = valeur_ptf_action.pct_change().dropna()
returns_ptf_oblig = valeur_ptf_oblig.pct_change().dropna()
returns_ptf_mon = valeur_ptf_mon.pct_change().dropna()

pilars_returns = pd.concat([returns_ptf_action, returns_ptf_oblig, returns_ptf_mon], axis = 1)
grid_test = Grid([0.07, 0.04, 0.025], pilars_returns, "daily", initial_grid)
optimized_grid = grid_test.optimize_grid(grid_test)
optimized_grid.plot_weights_evolution()
plt.show()
"""