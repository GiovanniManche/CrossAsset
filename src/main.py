import pandas as pd
from classes.Portfolio import Portfolio

data = pd.read_excel("data/Inputs.xlsx", sheet_name= "Actions").set_index("Date")
cols_with_na = [col for col in data.columns if data[col].isnull().sum() != 0]
data[cols_with_na] = data[cols_with_na].interpolate(method = 'linear')

x = Portfolio(data, "daily", "monthly", "discret")

pos_test = x.build_portfolio_v2()

x.plot_cumulative_returns()
pos = x.build_portfolio()
pos.to_excel("test positions Momentum.xlsx")
x.compute_portfolio_value(pos, 100).to_excel("Ptf value.xlsx")
