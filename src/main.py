import pandas as pd

from classes.Portfolio import Portfolio
from classes.Metrics import Metrics

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


"""
Première étape : Construction du pilier action
"""

# Choix d'exporter ou non les résutlats en excel
export_res:bool = False

# Import des données
data = pd.read_excel("data/Inputs v2.xlsx", sheet_name= "Actions").set_index("Date")
cols_with_na = [col for col in data.columns if data[col].isnull().sum() != 0]
data[cols_with_na] = data[cols_with_na].interpolate(method = 'linear')

# Instanciation du portefeuille
calculation_window:int = 6
list_strategy = ["momentum"]
list_weighting = ["ranking"]
ptf_action:Portfolio = Portfolio(data, periodicity = periodicity,
                                 rebalancement=rebalancing, method= method, list_strat=list_strategy, list_weighting=list_weighting,
                                 calculation_window=calculation_window, asset_class="equity")

# Backtest
ptf_action.run_backtest()

# Export des résultats
if export_res:
    ptf_action.positions.to_excel("test positions Momentum v3.xlsx")
    ptf_action.portfolio_value.to_excel("Valeur pilier action v3.xlsx")

# Affichage des métriques de performance du pilier action
bench: pd.DataFrame = ptf_action.benchmark.iloc[calculation_window:,].iloc[:,0]
metrics_action:Metrics = Metrics(ptf_action.portfolio_value, method =method, frequency=frequency, benchmark=bench)
print("Synthèse pour le pilier action : ")
premia_equity: float = metrics_action.synthesis()

"""
Troisième étape : Construction du pilier monétaire
"""
# Import des données
data_monetaire = pd.read_excel("data/Inputs v2.xlsx", sheet_name= "Monétaire").set_index("Date")
cols_with_na = [col for col in data_monetaire.columns if data_monetaire[col].isnull().sum() != 0]
data_monetaire[cols_with_na] = data_monetaire[cols_with_na].interpolate(method = 'linear')

# Sur le monétaire : pas de stratégie particulière
ptf_monetaire: pd.Series = data_monetaire

# Affichage des métriques de performance du pilier monétaire
metrics_monetaire:Metrics = Metrics(ptf_monetaire, method=method, frequency=frequency)
print("Statistiques pour le pilier monétaire : ")
metrics_monetaire.display_stats()

"""
Deuxième étape : Construction du pilier obligataire
"""
# Import des données
data_oblig = pd.read_excel("data/Inputs.xlsx", sheet_name= "Obligations").set_index("Date")
cols_with_na = [col for col in data_oblig.columns if data_oblig[col].isnull().sum() != 0]
data_oblig[cols_with_na] = data_oblig[cols_with_na].interpolate(method = 'linear')

# Instanciation du portefeuille
calculation_window:int = 6
list_strategy = ["momentum"]
list_weighting = ["ranking"]
ptf_obligataire:Portfolio = Portfolio(data_oblig, periodicity = periodicity,
                                 rebalancement=rebalancing, method= method, list_strat=list_strategy, list_weighting=list_weighting,
                                 calculation_window=calculation_window, asset_class="equity")

# Backtest sur l'obligataire
ptf_obligataire.run_backtest()

# Affichage des métriques de performance du pilier action
bench_oblig: pd.DataFrame = ptf_obligataire.benchmark.iloc[calculation_window:,].iloc[:,0]
metrics_obligation:Metrics = Metrics(ptf_obligataire.portfolio_value, method =method, frequency=frequency, benchmark = bench)
print("Sytnhèse pour le pilier obligataire : ")
bond_premia: float = metrics_obligation.synthesis()

"""
Quatrième étape : Export des rendements annuels pour déterminer
la prime d'un point de vue historique
"""

writer = pd.ExcelWriter('Cross Asset Salary.xlsm')
# A voir pour les rendements : est-ce qu'on coupe les x dernières valeurs ?
# Cas des données annuelles
if frequency != "daily":
    ret_action: pd.DataFrame = metrics_action.returns.iloc[:-1]
    ret_monetaire: pd.DataFrame = metrics_monetaire.returns.iloc[:-1]

# Export des résultats des trois portefeuilles
ret_action.to_excel(writer, sheet_name="Rendement annuel Portefeuille", startrow=2, startcol=2)

ret_monetaire.to_excel(writer, sheet_name="Rendement annuel Portefeuille", startrow=2, startcol=4)

a = 3
"""
Cinquième étape : Calcul du risque en CVaR de la grille initiale
"""



"""
Sixième étape : optimisation / construction de la nouvelle grille équilibrée et comparaison
"""

"""
Septième étape : Construction de la grille dynamique
"""

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