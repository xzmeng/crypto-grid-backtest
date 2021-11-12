
#%%
import grid
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from grid.util import cartesian_product

plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["figure.dpi"] = 100

df = grid.load_local_data('DOGEUSDT')
data = df.loc["2021-10":]
data.resample("1h").min().plot()
#%% backtest

params = {
    "step": 0.009,
    "cons2": 2,
    "cons3": 0.25,
    "cons4": 0.1,
}

backtest = grid.Backtest(data, **params)

backtest.run()
backtest.show_result()
backtest.plot()

#%% generate parameter combinations

params = {
    "step": 0.001,
    "cons2": 0.8,
    "cons3": 0.5,
    "cons4": 0.1,
}

step = np.arange(0.001, 0.01, 0.001)
cons2 = cons3 = cons4 = np.array([0.1, 0.25, 0.5, 1, 1.5, 2])

c = cartesian_product(step, cons2, cons3, cons4)

params_df = pd.DataFrame(c, columns=["step", "cons2", "cons3", "cons4"])
params_df = params_df.query('cons2 >= 1 and cons2 > cons3 and cons3 > cons4')

#%% optimize
params_list = params_df.to_dict(orient="records")
params_list

opt = grid.Optimization(data, params_list)
opt.run()

#%% analysis

opt.backtests.sort(key=lambda x: x.sharpe, reverse=True)
for backtest in opt.backtests:
    backtest.show_result()
