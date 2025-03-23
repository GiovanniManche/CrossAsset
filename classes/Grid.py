import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from scipy.optimize import minimize
from typing import Tuple, List, Dict, Optional, Union, Any
from concurrent.futures import ThreadPoolExecutor

class Grid:
    """
    Classe qui permet de créer une grille d'allocation d'actifs et lui associer différentes statistiques:
    rendements, variance, covariance, volatilité, corrélation.
    
    La grille représente une allocation d'actifs évoluant en fonction de l'horizon d'investissement.
    
    Attributs:
        weights (pd.DataFrame): Poids des actifs pour chaque horizon.
        hypothetical_returns (pd.DataFrame): Rendements hypothétiques de long terme.
        historical_returns (pd.DataFrame): Rendements historiques observés.
        expected_returns (pd.DataFrame): Rendements espérés (poids * rendements hypothétiques).
        returns_variance (pd.Series): Variance annualisée des rendements.
        returns_volatilities (pd.Series): Volatilité annualisée des rendements.
        returns_correlation (pd.DataFrame): Matrice de corrélation des rendements.
        returns_cov (pd.DataFrame): Matrice de covariance annualisée des rendements.
        portfolio_stats (pd.DataFrame): Statistiques du portefeuille (rendement, variance, volatilité).
        portfolio_variance (np.ndarray): Variance du portefeuille pour chaque horizon.
        portfolio_volatility (np.ndarray): Volatilité du portefeuille pour chaque horizon.
    """
    def __init__(self, grid_weights: pd.DataFrame, hypothetical_returns: pd.DataFrame, real_returns: pd.DataFrame, annalization_factor: int = 4) -> None:
        """
        Initialise la grille avec les poids, les rendements hypothétiques et les rendements réels.
        
        Args:
            grid_weights (pd.DataFrame): Poids des actifs pour chaque horizon d'investissement.
            hypothetical_returns (pd.DataFrame): Rendements hypothétiques de long terme pour chaque actif.
            real_returns (pd.DataFrame): Rendements historiques observés pour chaque actif.
            annalization_factor (int, optional): Facteur d'annualisation des rendements (4 pour trimestriel). Défaut: 4.
        
        Raises:
            ValueError: Si les noms de colonnes ou le nombre de colonnes ne correspondent pas entre les poids et les rendements.
        """
        if grid_weights.columns.tolist() != hypothetical_returns.columns.tolist():
            raise ValueError("Les noms des colonnes de poids et de rendement ne correspondent pas !")
        
        # Poids initiaux 
        self.weights: pd.DataFrame = grid_weights.copy()
        
        nb_col: int = self.weights.shape[1]
        if nb_col != hypothetical_returns.shape[1]:
            raise ValueError("Le nombre de colonne n'est pas identique entre les rendements et les parts !")
        
        # Rendements pour chaque horizon 
        self.hypothetical_returns: pd.DataFrame = hypothetical_returns.copy()
        self.historical_returns: pd.DataFrame = real_returns.copy()
        self.expected_returns: pd.DataFrame = self.weights * self.hypothetical_returns
        
        # Variance, covariance, corrélation des rendements historiques
        self.returns_variance: pd.Series = real_returns.var() * annalization_factor  
        self.returns_volatilities: pd.Series = real_returns.std() * np.sqrt(annalization_factor)  
        self.returns_correlation: pd.DataFrame = real_returns.corr()  
        self.returns_cov: pd.DataFrame = real_returns.cov() * annalization_factor 
        
        # Variance du portefeuille à chaque date = w.T * matrice cov * w
        portfolio_variance = np.zeros(self.weights.shape[0])
        for i, weights in enumerate(self.weights.values):
            portfolio_variance[i] = weights.T @ self.returns_cov @ weights
        
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Calcul du rendement total pour chaque date
        total_returns = self.expected_returns.sum(axis=1)
        
        # Dataframe avec les statistiques du portefeuille à chaque date
        self.portfolio_stats: pd.DataFrame = pd.DataFrame({
            'Rendement': total_returns,
            'Variance': pd.Series(portfolio_variance, index=self.weights.index),
            'Volatilité': pd.Series(portfolio_volatility, index=self.weights.index)
        })
        
        self.portfolio_variance: np.ndarray = portfolio_variance
        self.portfolio_volatility: np.ndarray = portfolio_volatility

    def calculate_gini(self, weights: np.ndarray) -> float:
        """
        Calcule le coefficient de Gini pour mesurer la concentration (1 = totalement concentré)
        
        Args:
            weights (np.ndarray): Vecteur des poids du portefeuille.
            
        Returns:
            float: Coefficient de Gini (entre 0 et 1).
        """
        # On trie les poids par ordre croissant
        sorted_weights = np.sort(weights)
        n = len(sorted_weights)
        
        # On calcule la somme cumulative des poids triés
        cum_weights = np.cumsum(sorted_weights)
        cum_weights_sum = cum_weights.sum()
        if cum_weights_sum == 0:  # Pour éviter la division par zéro
            return 0
        
        indices = np.arange(1, n + 1)
        gini = (2 * np.sum(indices * sorted_weights) / (n * cum_weights_sum)) - (n + 1) / n
        return gini

    def plot_weights_evolution(self, title: str = "Evolution de l'allocation en fonction du nombre d'années avant la retraite") -> plt.Axes:
        """
        Affiche l'évolution des poids du portefeuille sous forme de graphique à aires empilées.
        
        Args:
            title (str, optional): Titre du graphique. Défaut: "Evolution de l'allocation en fonction du nombre d'années avant la retraite".
            
        Returns:
            plt.Axes: L'objet Axes contenant le graphique.
        """
        plt.figure(figsize=(12, 6))

        ax = self.weights.plot.area(figsize=(12, 6), alpha=0.7, stacked=True)

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

    def plot_risk_return(self, title: str = "Evolution du couple risque/rendement en fonction du nombre d'années avant la retraite") -> plt.Figure:
        """
        Affiche l'évolution du rendement et du risque en fonction de l'horizon d'investissement.
        
        Args:
            title (str, optional): Titre du graphique. Défaut: "Evolution du couple risque/rendement en fonction du nombre d'années avant la retraite".
            
        Returns:
            plt.Figure: L'objet Figure contenant le graphique.
        """
        fig, ax = plt.subplots(figsize=(12, 6))
    
        # On trace la courbe de vol et la courbe de rendement
        ax.plot(self.portfolio_stats.index, self.portfolio_stats['Rendement'], 
            '-o', color='darkblue', linewidth=2, markersize=5, label='Rendement espéré')
        ax.plot(self.portfolio_stats.index, self.portfolio_stats['Volatilité'], 
            '-o', color='darkred', linewidth=2, markersize=5, label='Volatilité')
    
        # Personnalisation 
        ax.set_xlabel('Horizon d\'investissement (années)', fontsize=12)
        ax.set_ylabel('Pourcentage (%)', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.3)
    
        # On inverse l'axe x pour avoir l'horizon de 40 à 0
        ax.invert_xaxis()

        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        ax.legend(loc='best')
        plt.tight_layout()
        return fig

    def plot_correlation_heatmap(self, title: str = "Matrice de corrélation des rendements historiques", annot: bool = False) -> plt.Axes:
        """
        Affiche la matrice de corrélation entre les rendements des actifs.
        
        Args:
            title (str, optional): Titre du graphique. Défaut: "Matrice de corrélation des rendements historiques".
            annot (bool, optional): Si True, affiche les valeurs numériques dans la heatmap. Défaut: False.
            
        Returns:
            plt.Axes: L'objet Axes contenant le graphique.
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(self.returns_correlation, annot=annot, cmap='coolwarm', 
                vmin=-1, vmax=1, center=0, fmt='.2f', linewidths=0.5)
        plt.title(title)
        plt.tight_layout()
        return plt.gca()

    def plot_gini_evolution(self, title: str = "Evolution du coefficient de Gini (concentration des actifs)") -> plt.Figure:
        """
        Affiche l'évolution du coefficient de Gini en fonction de l'horizon d'investissement.
        
        Args:
            title (str, optional): Titre du graphique. Défaut: "Evolution du coefficient de Gini (concentration des actifs)".
            
        Returns:
            plt.Figure: L'objet Figure contenant le graphique.
        """
        # Calcul du coefficient de Gini pour chaque horizon
        gini_coefficients = [self.calculate_gini(weights) for weights in self.weights.values]
        gini_series = pd.Series(gini_coefficients, index=self.weights.index)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(gini_series.index, gini_series, 
                '-o', color='purple', linewidth=2, markersize=5, label='Coefficient de Gini')
        
        ax.set_xlabel('Horizon d\'investissement (années)', fontsize=12)
        ax.set_ylabel('Coefficient de Gini', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.invert_xaxis()
        
        # Ajout de lignes horizontales pour les repères
        ax.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, label='Faible concentration')
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Concentration moyenne')
        ax.axhline(y=0.7, color='gray', linestyle='--', alpha=0.5, label='Forte concentration')
        
        ax.legend(loc='best')
        plt.tight_layout()
        return fig

    def optimize_grid(self, initial_grid: 'Grid', max_gini: float = 0.6) -> 'Grid':
        """
        Optimise la grille pour maximiser le rendement tout en respectant la contrainte
        de volatilité (pas plus que la volatilité initiale) et de diversification (coefficient de Gini).
        
        Args:
            initial_grid (Grid): La grille initiale dont on veut respecter les contraintes de volatilité.
            max_gini (float, optional): Coefficient de Gini maximal autorisé (0-1). Défaut: 0.6.
            
        Returns:
            Grid: Une nouvelle instance de Grid avec les poids optimisés.
        """
        n_assets: int = self.weights.shape[1]
        horizons = initial_grid.weights.index
        target_volatilities = initial_grid.portfolio_stats["Volatilité"]
        optimized_weights = pd.DataFrame(
            index=horizons,
            columns=self.weights.columns,
            dtype=float
            )
        
        optimized_stats = pd.DataFrame(
            index=horizons,
            columns=['Rendement', 'Volatilité', 'Gini'],
            dtype=float
            )

        bounds = [(-0.25,1) for _ in range(n_assets)]

        def optimize_single_horizon(horizon: Union[int, str]) -> Tuple[Union[int, str], np.ndarray, float, float, float, bool]:
            """
            Optimise l'allocation pour un horizon spécifique.
            
            Args:
                horizon (int ou str): L'horizon d'investissement.
                
            Returns:
                tuple: (horizon, poids optimisés, rendement, volatilité, gini, succès)
            """
            target_volatility = target_volatilities.loc[horizon]
            initial_weights = self.weights.loc[horizon].values

            def objective_function(weights: np.ndarray) -> float:
                """
                Fonction objectif à minimiser (rendement négatif pour maximisation).
                
                Args:
                    weights (np.ndarray): Vecteur des poids du portefeuille.
                    
                Returns:
                    float: Rendement négatif attendu.
                """
                # On multiplie par -1 car on cherche à maximiser le rendement
                return -np.dot(weights, self.hypothetical_returns.loc[horizon].values)
            
            def gini_constraint(weights: np.ndarray) -> float:
                """
                Contrainte pour limiter la concentration du portefeuille.
                
                Args:
                    weights (np.ndarray): Vecteur des poids du portefeuille.
                    
                Returns:
                    float: Différence entre le Gini maximum autorisé et le Gini calculé.
                    Valeur positive si la contrainte est respectée.
                """
                return max_gini - self.calculate_gini(weights)
            
            constraints = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1.3},  # somme des poids = 1
                # Vol de la nouvelle grille au pire égale à celle de l'ancienne
                {"type": "ineq", "fun": lambda w: target_volatility - np.sqrt(w @ self.returns_cov @ w)},
                # Contrainte sur le coefficient de Gini (diversification)
                {"type": "ineq", "fun": gini_constraint}
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
                new_returns = np.dot(new_weights, self.hypothetical_returns.loc[horizon].values)
                new_vol = np.sqrt(new_weights @ self.returns_cov @ new_weights)
                new_gini = self.calculate_gini(new_weights)
                return horizon, new_weights, new_returns, new_vol, new_gini, True
            
            else:
                print(f"L'optimisation a échoué pour l'horizon {horizon}. Message: {result.message}")
                # Utiliser les poids initiaux en cas d'échec
                fake_weights = initial_weights
                ret = np.dot(fake_weights, self.hypothetical_returns.loc[horizon].values)
                vol = np.sqrt(fake_weights @ self.returns_cov @ fake_weights)
                gini = self.calculate_gini(fake_weights)
                return horizon, fake_weights, ret, vol, gini, False

        # Utiliser un ThreadPoolExecutor pour paralléliser les optimisations
        with ThreadPoolExecutor(max_workers=min(10, len(horizons))) as executor:
            results = list(executor.map(optimize_single_horizon, horizons))
    
        # Traiter les résultats
        for horizon, weights, ret, vol, gini, success in results:
            optimized_weights.loc[horizon] = weights
            optimized_stats.loc[horizon, 'Rendement'] = ret
            optimized_stats.loc[horizon, 'Volatilité'] = vol
            optimized_stats.loc[horizon, 'Gini'] = gini
    
        # Création de la nouvelle grille optimisée avec les statistiques complètes
        optimized_grid = Grid(optimized_weights, self.hypothetical_returns, self.historical_returns, 4)
    
        # Ajout des coefficients de Gini à la grille optimisée
        optimized_grid.portfolio_stats['Gini'] = optimized_stats['Gini']
    
        # Résumer les résultats
        successful = sum(1 for _, _, _, _, _, success in results if success)
        print(f"Optimisation réussie pour {successful} horizons sur {len(horizons)}")
        
        return optimized_grid

    def compare_with(self, other_grid: 'Grid', title: str = "Comparaison des grilles") -> plt.Figure:
        """
        Compare cette grille avec une autre grille (notamment avant/après optimisation).
        
        Args:
            other_grid (Grid): Une autre instance de Grid à comparer avec celle-ci.
            title (str, optional): Titre du graphique. Défaut: "Comparaison des grilles".
            
        Returns:
            plt.Figure: L'objet Figure contenant le graphique de comparaison.
        """
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        
        # Comparaison des rendements
        ax1.plot(self.portfolio_stats.index, self.portfolio_stats['Rendement'], 
                '-o', color='darkblue', linewidth=2, markersize=5, label='Grille originale')
        ax1.plot(other_grid.portfolio_stats.index, other_grid.portfolio_stats['Rendement'], 
                '-o', color='royalblue', linewidth=2, markersize=5, label='Grille optimisée')
        ax1.set_ylabel('Rendement attendu', fontsize=12)
        ax1.set_title(f"{title} - Rendements", fontsize=14)
        ax1.grid(True, linestyle='--', alpha=0.3)
        ax1.legend(loc='best')
        ax1.invert_xaxis()
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        
        # Comparaison des volatilités
        ax2.plot(self.portfolio_stats.index, self.portfolio_stats['Volatilité'], 
                '-o', color='darkred', linewidth=2, markersize=5, label='Grille originale')
        ax2.plot(other_grid.portfolio_stats.index, other_grid.portfolio_stats['Volatilité'], 
                '-o', color='salmon', linewidth=2, markersize=5, label='Grille optimisée')
        ax2.set_ylabel('Volatilité', fontsize=12)
        ax2.set_title(f"{title} - Volatilité", fontsize=14)
        ax2.grid(True, linestyle='--', alpha=0.3)
        ax2.legend(loc='best')
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        
        # Comparaison des coefficients de Gini
        original_gini = [self.calculate_gini(weights) for weights in self.weights.values]
        optimized_gini = [other_grid.calculate_gini(weights) for weights in other_grid.weights.values]
        
        ax3.plot(self.weights.index, original_gini, 
                '-o', color='purple', linewidth=2, markersize=5, label='Grille originale')
        ax3.plot(other_grid.weights.index, optimized_gini, 
                '-o', color='orchid', linewidth=2, markersize=5, label='Grille optimisée')
        ax3.set_xlabel('Horizon d\'investissement (années)', fontsize=12)
        ax3.set_ylabel('Coefficient de Gini', fontsize=12)
        ax3.set_title(f"{title} - Diversification (Gini)", fontsize=14)
        ax3.grid(True, linestyle='--', alpha=0.3)
        ax3.legend(loc='best')
        
        plt.tight_layout()
        return fig