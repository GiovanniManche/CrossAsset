import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, Dict
from classes.grid import Grid

class Visualisation:
    """
    Cette classe sert exclusivement à centraliser les éléments d'affichages graphiques.
    """
    def __init__(self):
        pass

    @staticmethod
    def heatmap_correlation(returns: pd.DataFrame,annot: bool = True, cmap: str = 'coolwarm', 
                            vmin: float = -1, vmax: float = 1,
                            title: str = "Matrice de corrélation") -> None:
        """
        Affiche la matrice de corrélation des rendements 
        """
        correlations = returns.corr()
        plt.figure(figsize=(12, 6))

        heatmap = sns.heatmap(
            correlations,
            annot=annot,
            fmt=".2f",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
        )
        plt.title(title, fontsize=16, pad=20)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_cumulative_returns(returns: pd.DataFrame, 
                                title: str = "Rendements cumulés") -> None:
        """
        Calcule et affiche les rendements cumulés pour chaque actif.
        """
        cumulative_returns: pd.DataFrame = (1 + returns).cumprod()
        cumulative_returns.plot()
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Valeur")
        plt.legend(title="Actifs", frameon=True, fontsize=8)
        plt.axhline(y=1, color="red", linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_weights(weights: pd.DataFrame, title="Évolution des poids du portefeuille",
                     invert_xaxis: bool = False) -> plt.Axes:
        if weights.empty:
            raise ValueError("Les positions du portefeuille n'ont pas été calculées. Exécutez d'abord run_backtest().")
    
        # Création du graphique à aire empilée
        ax = weights.plot.area(figsize=(12, 6), alpha=0.7, stacked=True)
    
        # Si on veut inverser l'axe des abscisses ==> grille de désensibilisation
        if invert_xaxis:
            ax.invert_xaxis()
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Allocation (%)', fontsize=12)
        ax.legend(title='Actifs', loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle='--', alpha=0.7)
        for y in np.arange(0.2, 1.0, 0.2):
            ax.axhline(y=y, color='gray', linestyle='-', alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        plt.tight_layout()
        return ax

    @staticmethod
    def plot_grid_cumulative_returns_grid(grid:Grid, compare_with: Optional[Grid] = None, 
                           title: str = "Rendements cumulés",
                           figsize: Tuple[int, int] = (12, 6)) -> plt.Axes:
        """
        Affiche les rendements cumulés d'une grille.
        """
        fig, ax = plt.subplots(figsize=figsize)
    
        # Calcul des rendements cumulés
        cumulative_returns = (1 + grid.grid_returns).cumprod()
        cumulative_returns.plot(ax=ax, linewidth=2, label="Grille actuelle")
    
        # Si une grille de comparaison est fournie, on ajoute ses rendements cumulés
        if compare_with is not None:
            compare_cumulative_returns = (1 + compare_with.grid_returns).cumprod()
            compare_cumulative_returns.plot(ax=ax, linewidth=2, linestyle='--', label="Grille comparée")
    
        # On inverse l'axe des abscisses
        ax.invert_xaxis()
    
        # Ajout des éléments de mise en forme
        ax.set_title(f"{title} (rendements {grid.periodicity})", fontsize=14)
        ax.set_xlabel("Horizon avant la retraite (années)", fontsize=12)
        ax.set_ylabel("Valeur", fontsize=12)
        ax.legend(title="Grilles", frameon=True, fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))
        ax.axhline(y=1, color="red", linestyle="--", alpha=0.3)
        ax.grid(True, linestyle='--', alpha=0.7)
    
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.2f}'.format(y)))
    
        plt.tight_layout()
        return ax
    
    @staticmethod
    def plot_CVaR(grid: Grid, compare_with: Optional[Grid] = None, 
              title: str = "Évolution de la CVaR en fonction de l'horizon", 
              figsize: Tuple[int, int] = (12, 6)) -> plt.Axes:
        
        """
        Affichage du graphique de la CVaR d'une grille
        """
    
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(grid.grid_CVaR.index, grid.grid_CVaR.values, linewidth=2, label="Grille actuelle")
    
        if compare_with is not None:
            ax.plot(compare_with.grid_CVaR.index, compare_with.grid_CVaR.values, 
                linewidth=2, linestyle='--', label="Grille comparée")
    
        ax.invert_xaxis()
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Horizon avant la retraite (années)", fontsize=12)
        ax.set_ylabel("CVaR", fontsize=12)
        ax.legend(title="Grilles", frameon=True, fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.axhline(y=0, color="red", linestyle="--", alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.2%}'.format(y) if abs(y) < 1 else '{:.2f}'.format(y)))
    
        plt.tight_layout()
        return ax

    @staticmethod
    def plot_cumulative_returns(returns_dict: Dict[str, pd.Series], 
                            title: str = "Rendements cumulés",
                            periodicity: str = "yearly",
                            figsize: Tuple[int, int] = (12, 6),
                            line_styles: Optional[Dict[str, str]] = None,
                            colors: Optional[Dict[str, str]] = None) -> plt.Axes:
        """
        Affiche les rendements cumulés pour un ensemble de séries de rendements.
        """
        fig, ax = plt.subplots(figsize=figsize)
    
        # Styles de ligne par défaut si aucun n'est fourni
        if line_styles is None:
            line_styles = {}
    
        # Couleurs par défaut si aucune n'est fournie
        if colors is None:
            colors = {}
        
        for i, (name, returns) in enumerate(returns_dict.items()):
            cumulative_returns = (1 + returns).cumprod()
            line_style = line_styles.get(name, "-" if i == 0 else "--")
            color = colors.get(name, None)  
            cumulative_returns.plot(ax=ax, linewidth=2, label=name, 
                              linestyle=line_style, color=color)
    
    
        # Ajout des éléments de mise en forme
        ax.set_title(f"{title} (rendements {periodicity})", fontsize=14)
        ax.set_xlabel("Date", fontsize=12)
    
        ax.set_ylabel("Valeur", fontsize=12)
        ax.legend(title="Séries", frameon=True, fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))
        ax.axhline(y=1, color="red", linestyle="--", alpha=0.3)
        ax.grid(True, linestyle='--', alpha=0.7)
    
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.2f}'.format(y)))
    
        plt.tight_layout()
        return ax
