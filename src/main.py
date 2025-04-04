import pandas as pd, numpy as np 
import matplotlib.pyplot as plt
from classes.portfolio import Portfolio
from classes.visualisation import Visualisation
from classes.metrics import Metrics
from classes.pilar import Pilar
from classes.grid import Grid
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)  
warnings.filterwarnings('ignore', category=DeprecationWarning) 

"""
Main permettant de construire une grille de désensibilisation par optimisation
sur la CVAR
"""

"""
Définition des paramètres pour le backtester
"""
periodicity: str = "monthly"
rebalancing: str = "quarterly"
frequency: str = "yearly"
method: str = "discret"
calculation_window:int = 6

"""
Première étape : Construction du pilier action
"""
# Choix d'exporter ou non les résutlats en excel
export_res:bool = False

# Import des données
core_stock_indices = ["SPX Index", "SX5E Index", "NKY Index", "MXEF Index"]
all_stock_prices = pd.read_excel("data/Inputs.xlsx", sheet_name= "Actions").set_index("Date")
# Gestion des NA par interpolation
cols_with_na = [col for col in all_stock_prices.columns if all_stock_prices[col].isnull().sum() != 0]
all_stock_prices[cols_with_na] = all_stock_prices[cols_with_na].interpolate(method = 'linear')


# Construction de la poche Momentum
core_stock_prices = all_stock_prices[core_stock_indices]
not_core_prices = all_stock_prices.loc[:, ~all_stock_prices.columns.isin(core_stock_indices)]
list_strategy = ["momentum"]
list_weighting = ["ranking"]
momentum_stock_portfolio = Portfolio(not_core_prices, periodicity = periodicity,
                                 rebalancement= rebalancing, method="discret", list_strat=list_strategy, list_weighting=list_weighting,
                                 calculation_window=6, asset_class="equity")

# Backtest
momentum_stock_portfolio.run_backtest()
Visualisation.plot_weights(momentum_stock_portfolio.positions, 
                           title = "Evolution des poids, poche Momentum - Actions")
Visualisation.heatmap_correlation(momentum_stock_portfolio.returns["monthly"])

# Export des résultats
if export_res:
    momentum_stock_portfolio.positions.to_excel("test positions Momentum v3.xlsx")
    momentum_stock_portfolio.portfolio_value.to_excel("Valeur pilier action v3.xlsx")

# Affichage des métriques de performance du pilier action
bench: pd.DataFrame = momentum_stock_portfolio.benchmark.iloc[calculation_window:,].iloc[:,0]
metrics_action:Metrics = Metrics(momentum_stock_portfolio.portfolio_value, 
                                 method =method, frequency=frequency, benchmark=bench)
print("Synthèse pour le portefeuille momentum action : ")
premia_equity: float = metrics_action.synthesis()

# Répartition core - Momentum
momentum_stock_portfolio.portfolio_value.name = "Momentum Ptf Value"
# Rmq : comme momentum_stock_portfolio.portfolio_value est déjà en monthly, 
# pilar_member_prices va automatiquement récupérer les prix mensuels aux bonnes dates. 
pilar_member_prices = core_stock_prices.join(momentum_stock_portfolio.portfolio_value, how='inner')
stock_pilar = Pilar(asset_prices=pilar_member_prices,
                    periodicity="monthly", method="discret")
minimal_weights_constraints = {
    "SPX Index": 0.2,
    "SX5E Index": 0.25,
    "NKY Index": 0.05,
    "MXEF Index": 0.05, 
    "Momentum Ptf Value": 0.15
}
# Calcul des parts
stock_weights, stock_returns = stock_pilar.run_min_variance_optimization(
    calculation_window=calculation_window, rebalancing_freq=rebalancing,
    min_weights=minimal_weights_constraints)
Visualisation.plot_weights(stock_weights)
Visualisation.heatmap_correlation(stock_pilar.returns)

# Métriques
stock_pilar_value = stock_pilar.compute_pilar_value(stock_weights)
metrics_action:Metrics = Metrics(stock_pilar_value, 
                                 method =method, frequency=frequency, benchmark=bench)
print("Synthèse pour le pilier action : ")
premia_equity: float = metrics_action.synthesis()

"""
Deuxième étape : Construction du pilier obligataire
"""
# Import des données
core_bond_indices = ["LBEATREU Index", "LBUSTRUU Index", "LEGATRUU Index"]
all_bond_prices = pd.read_excel("data/Inputs.xlsx", sheet_name= "Obligataire").set_index("Date")
## Gestion des NA par interpolation
cols_with_na = [col for col in all_bond_prices.columns if all_bond_prices[col].isnull().sum() != 0]
all_bond_prices[cols_with_na] = all_bond_prices[cols_with_na].interpolate(method = 'linear')

# Construction de la poche Momentum
core_bond_prices = all_bond_prices[core_bond_indices]
not_core_bond_prices = all_bond_prices.loc[:, ~all_bond_prices.columns.isin(core_bond_indices)]
list_strategy = ["momentum"]
list_weighting = ["ranking"]
momentum_bond_portfolio = Portfolio(not_core_bond_prices, periodicity = periodicity,
                                 rebalancement= rebalancing, method="discret", list_strat=list_strategy, list_weighting=list_weighting,
                                 calculation_window=6, asset_class="bonds")

## Backtest
momentum_bond_portfolio.run_backtest()
Visualisation.plot_weights(momentum_bond_portfolio.positions, 
                           title = "Evolution des poids, poche Momentum - Obligations")
Visualisation.heatmap_correlation(momentum_bond_portfolio.returns["monthly"])

## Métriques 
bench: pd.DataFrame = momentum_bond_portfolio.benchmark.iloc[calculation_window:,].iloc[:,0]
metrics_bond:Metrics = Metrics(momentum_bond_portfolio.portfolio_value, 
                               method =method, frequency=frequency, benchmark=bench)
print("Synthèse pour le portefeuille momentum obligations : ")
premia_bond: float = metrics_bond.synthesis()

# Répartition core - momentum
momentum_bond_portfolio.portfolio_value.name = "Momentum Ptf Value"
# Rmq : comme momentum_bond_portfolio.portfolio_value est déjà en monthly, 
# pilar_member_prices va automatiquement récupérer les prix mensuels aux bonnes dates. 
bond_pilar_member_prices = core_bond_prices.join(momentum_bond_portfolio.portfolio_value, how='inner')
bond_pilar = Pilar(asset_prices=bond_pilar_member_prices,
                   periodicity="monthly", method="discret")
minimal_weights_constraints = {
    "LBEATREU Index": 0.2,
    "LBUSTRUU Index": 0.25,
    "LEGATRUU Index": 0.05, 
    "Momentum Ptf Value": 0.15
}

## Calcul des parts
bond_weights, bond_returns = bond_pilar.run_min_variance_optimization(
    calculation_window=calculation_window, rebalancing_freq=rebalancing,
    min_weights=minimal_weights_constraints)
Visualisation.plot_weights(bond_weights)
Visualisation.heatmap_correlation(bond_pilar.returns)

## Métriques 
bond_pilar_value = bond_pilar.compute_pilar_value(bond_weights)
metrics_bond:Metrics = Metrics(bond_pilar_value, 
                               method=method, frequency=frequency, benchmark=bench)
print("Synthèse pour le pilier obligation : ")
premia_bond: float = metrics_bond.synthesis()


"""
Troisième étape : Construction du pilier monétaire
"""
# Import des données
data_monetaire = pd.read_excel("data/Inputs.xlsx", sheet_name= "Monétaire").set_index("Date")
cols_with_na = [col for col in data_monetaire.columns if data_monetaire[col].isnull().sum() != 0]
data_monetaire[cols_with_na] = data_monetaire[cols_with_na].interpolate(method = 'linear')

# Sur le monétaire : pas de stratégie particulière
ptf_monetaire: pd.Series = data_monetaire

# Affichage des métriques de performance du pilier monétaire
metrics_monetaire:Metrics = Metrics(ptf_monetaire, method=method, frequency=frequency)
print("Statistiques pour le pilier monétaire : ")
metrics_monetaire.display_stats()

"""
Quatrième étape : Export des rendements annuels pour déterminer
la prime d'un point de vue historique
"""

writer = pd.ExcelWriter('Cross Asset Salary.xlsm')
# A voir pour les rendements : est-ce qu'on coupe les x dernières valeurs ?
# Cas des données annuelles
if frequency == "yearly":
    ret_action: pd.DataFrame = metrics_action.returns.iloc[:-1]
    ret_monetaire: pd.DataFrame = metrics_monetaire.returns.iloc[:-1]

# Export des résultats des trois portefeuilles
ret_action.to_excel(writer, sheet_name="Rendement annuel Portefeuille", startrow=2, startcol=2)

ret_monetaire.to_excel(writer, sheet_name="Rendement annuel Portefeuille", startrow=2, startcol=4)

a = 3
"""
Cinquième étape : Calcul du risque en CVaR de la grille initiale
"""
initial_grid_weights = pd.read_excel("data/Inputs.xlsx", sheet_name= "Grille initiale").set_index("Horizon")
# On calcule les rendements via la méthode de Pilar
initial_assets = pd.concat([all_stock_prices["MSDEE15N Index"], all_bond_prices["LBEATREU Index"], data_monetaire["OISESTR Index"]], axis = 1).dropna()
initial_returns = Pilar(asset_prices=initial_assets, periodicity=periodicity, method = method).compute_asset_returns(periodicity)
initial_grid = Grid(
    long_term_returns=[0.06, 0.035, 0.02],
    portfolios_returns=initial_returns,
    periodicity=periodicity, 
    grid_weights=initial_grid_weights
)
Visualisation.plot_weights(initial_grid.grid_weights, 
                           title = "Evolution des poids de la grille initiale",
                           invert_xaxis= True)
Visualisation.plot_CVaR(grid = initial_grid,
                        title = "Evolution de la CVaR de la grille équilibre initiale")


"""
Sixième étape : optimisation / construction de la nouvelle grille équilibrée et comparaison
"""
# Récupération des rendements
# Travail nécessaire pour monétaire car plus de rendements que les autres
new_pilars_returns = pd.concat([stock_returns, bond_returns], axis = 1)
cash_returns = Metrics(ptf_monetaire, method=method, frequency="monthly").returns
rows_to_drop = len(cash_returns) - len(new_pilars_returns)
trimmed_cash_returns = cash_returns.iloc[rows_to_drop:].copy()
trimmed_cash_returns = trimmed_cash_returns.reset_index(drop = True)
trimmed_cash_returns.index = new_pilars_returns.index
new_pilars_returns = pd.concat([new_pilars_returns, trimmed_cash_returns], axis = 1)
new_pilars_returns.columns = ["Actions", "Obligations", "Monétaire"]

## Optimisation par maximisation du rendement sous contrainte de CVaR
### ATTENTION, CHANGER LES HYP DE RDT LT !!!!!!!!!!!!!!!!
new_grid = Grid([0.07, 0.04, 0.035], new_pilars_returns, "monthly", initial_grid.grid_weights)
new_grid = new_grid.optimize_grid(initial_grid)
Visualisation.plot_weights(new_grid.grid_weights, title="Evolution des poids - Grille équilibre optimisée",
                           invert_xaxis=True)
Visualisation.plot_CVaR(new_grid, compare_with=initial_grid,
                        title="Evolution de la CVaR : comparaison entre la grille optimisée (actuelle) et la grille initiale")

"""
Septième étape : Construction de la grille dynamique
"""
initial_dynamic_grid_weights = pd.read_excel("data/Inputs.xlsx", sheet_name= "Grille dynamique").set_index("Horizon")
initial_dynamic_grid = Grid(
    long_term_returns=[0.06, 0.035, 0.02],
    portfolios_returns=initial_returns,
    periodicity=periodicity, 
    grid_weights=initial_dynamic_grid_weights
)
Visualisation.plot_weights(initial_dynamic_grid.grid_weights, 
                           title = "Evolution des poids de la grille dynamique initiale",
                           invert_xaxis= True)
Visualisation.plot_CVaR(grid = initial_dynamic_grid, compare_with=initial_grid,
                        title = "Evolution de la CVaR : comparaison entre grille équilibre initiale et grille dynamique")

plt.show()
"""
old

pos_test = x.build_portfolio_v2()
x.compute_portfolio_value_v2(100, calculation_window)

# Affichage des métriques de performance du pilier action
metrics_action:Metrics = Metrics(x.portfolio_value, method ="discret", frequency="daily")
print("Statistiques pour le pilier action : ")
metrics_action.display_stats()

# Comparaison avec le MSCI World (Benchmark)
# Retraitement du bench pour conserver la même périodicité
bench: pd.DataFrame = x.benchmark.iloc[calculation_window:,]
bench:pd.Series = bench.iloc[:,0]
metrics_bench: Metrics = Metrics(bench, method = "discret", frequency = "daily")
print("Statistiques pour le benchmark : ")
metrics_bench.display_stats()

# Prime de diversification
action_premia:float = metrics_action.compute_sharpe_ratio() - metrics_bench.compute_sharpe_ratio()
print(f'La prime de diversification du pilier action est de {round(action_premia,4)} pour chaque point de volatilité supplémentaire')

# Comparaison avec le MSCI World (Benchmark)
# Retraitement du bench pour conserver la même périodicité
bench: pd.DataFrame = ptf_action.benchmark.iloc[calculation_window:,]
bench:pd.Series = bench.iloc[:,0]
metrics_bench: Metrics = Metrics(bench, method = "discret", frequency = "daily")
print("Statistiques pour le benchmark : ")
metrics_bench.display_stats()

# Comparaison avec le benchmark
# Retraitement du bench pour conserver la même périodicité
bench_oblig: pd.DataFrame = ptf_obligataire.benchmark.iloc[calculation_window:,]
bench_oblig:pd.Series = bench.iloc[:,0]
metrics_bench_oblig: Metrics = Metrics(bench_oblig, method = "discret", frequency = "daily")
print("Statistiques pour le benchmark obligataire : ")
metrics_bench_oblig.display_stats()

# Prime de diversification
bond_premia:float = metrics_action.compute_sharpe_ratio() - metrics_bench.compute_sharpe_ratio()
print(f'La prime de diversification du pilier obligataire est de {round(bond_premia,4)} pour chaque point de volatilité supplémentaire')


# Prime de diversification
action_premia:float = metrics_action.compute_sharpe_ratio() - metrics_bench.compute_sharpe_ratio()
print(f'La prime de diversification du pilier action est de {round(action_premia,4)} pour chaque point de volatilité supplémentaire')


x.plot_cumulative_returns()
pos = x.build_portfolio()
pos.to_excel("test positions Momentum.xlsx")
x.compute_portfolio_value(pos, 100).to_excel("Ptf value.xlsx")
"""