# for debug
if __name__ == "__main__":
    import sys
    from os.path import dirname

    sys.path.insert(0, dirname(dirname(__file__)))
    from grid.data import load_local_data

import os
import pandas as pd
from os.path import dirname, join

BACKTEST_DATA_PATH = join(dirname(dirname(__file__)), "data/backtest")


class Backtest:
    def __init__(
        self, data, step, cons2, cons3, cons4, size_usdt=100, commission=0.0002
    ):
        self.orig_data = data
        self.step = step
        self.size_usdt = size_usdt
        self.commission = commission

        self.cons2 = cons2
        self.cons3 = cons3
        self.cons4 = cons4

        self.mid = self.orig_data.price[0]
        self.low = self.mid * (1 - self.step)
        self.high = self.mid * (1 + self.step)

    def grid_up(self):
        self.low = self.mid
        self.mid = self.high
        self.high = self.mid * (1 + self.step)

    def grid_down(self):
        self.high = self.mid
        self.mid = self.low
        self.low = self.mid * (1 - self.step)

    def run(self):
        data = self.orig_data.copy()
        buy_indices = []
        sell_indices = []

        for i, price in enumerate(data.price):
            if price < self.low:
                buy_indices.append(i)
                self.grid_down()
            if price > self.high:
                sell_indices.append(i)
                self.grid_up()

        data["buy"] = 0
        data.iloc[buy_indices, data.columns.get_loc("buy")] = 1
        data.iloc[sell_indices, data.columns.get_loc("buy")] = -1
        data = data.loc[data.buy != 0].copy()

        data["buy_cumsum"] = data.buy.cumsum()
        data["size_usdt"] = self.size_usdt

        y = data.buy[data.buy != 0]
        data["cons_buy"] = 0
        data.loc[y.index, "cons_buy"] = (
            y.groupby((y != y.shift()).cumsum()).cumcount() + 1
        )

        data.loc[data.cons_buy >= 2, "size_usdt"] = self.size_usdt * self.cons2
        data.loc[data.cons_buy >= 3, "size_usdt"] = self.size_usdt * self.cons3
        data.loc[data.cons_buy >= 4, "size_usdt"] = self.size_usdt * self.cons4

        data["usdt_change"] = data.buy * data.size_usdt * (-1)
        data["coin_change"] = data.usdt_change / data.price * (-1)
        data["usdt"] = data.usdt_change.cumsum()
        data["coin"] = data.coin_change.cumsum()
        data["coin_value"] = data.coin * data.price
        data["comm"] = data.usdt_change * self.commission
        data["total_comm"] = data.comm.cumsum()
        data["profit"] = data.usdt + data.coin_value - data.total_comm
        self.data = data.round(
            {
                "usdt_change": 2,
                "coin_change": 2,
                "usdt": 2,
                "coin": 2,
                "coin_value": 2,
                "comm": 4,
                "total_comm": 4,
                "profit": 2,
            }
        )

    def save(self):
        os.makedirs(BACKTEST_DATA_PATH, exist_ok=True)
        filename = join(BACKTEST_DATA_PATH, f"{self.params_text}.csv")
        print(f'Saving {filename}...')
        self.data.to_csv(filename)

    @property
    def params_text(self):
        text = (
            f"step={self.step},cons2={self.cons2},cons3={self.cons3},"
            f"cons4={self.cons4}"
        )
        return text

    def show_result(self):
        data = self.data
        start = data.price.index[0]
        end = data.price.index[-1]
        price = data.price
        print(f"----{self.params_text}----")
        print(f"{(end - start).days} days from {start} to {end}")
        print(
            f"price - start: {price[0]}, end: {price[-1]}, max: {price.max()}, "
            f"min: {price.min()}"
        )
        print(
            f"min profit: {data.profit.min()} ({data.index[data.profit.argmin()]}), "
            f"max profit: {data.profit.max()} ({data.index[data.profit.argmax()]}), "
            f"end profit: {data.profit[-1]}"
        )
        print(
            f"trade count: {len(data.buy[data.buy != 0])}, max buy_cumsum: {data.buy_cumsum.max()}, "
            f"min buy_cumsum: {data.buy_cumsum.min()}, end buy_cumsum: {data.buy_cumsum[-1]}"
        )

        cons_percent = data.cons_buy.value_counts(normalize=True)
        print(
            f"cons_buy percentage - cons1: {cons_percent[1]:.1%}, cons2: {cons_percent[2]:.1%}, "
            f"cons3: {cons_percent[3]:.1%}, cons4: {cons_percent[4]:.1%}, "
            f">= 5: {cons_percent.loc[5:].sum():.1%}"
        )
        print(f"sharpe: {self.sharpe}")
        print()

    def plot(self):
        data = self.data
        data[["price", "profit"]].resample("1h").min().plot(secondary_y="profit")

    
    @property
    def profit_per_loss(self):
        ratio = 0
        min_profit = self.data.profit.min()    
        end_profit = self.data.profit[-1]
        if end_profit > 0 and min_profit < 0:
            return end_profit / (- min_profit)
        if min_profit > 0:
            return 10e6
        if end_profit < 0:
            return -10e6


    # calculate the sharpe ratio
    @property
    def sharpe(self):
        data = self.data
        return data.profit.mean() / data.profit.std()


class Optimization:
    def __init__(self, data, params_list):
        self.backtests = [Backtest(data, **params) for params in params_list]

    def run(self):
        for i, backtest in enumerate(self.backtests, start=1):
            print(f'[{i}/{len(self.backtests)}]')
            backtest.run()
            backtest.show_result()

    def plot(self):
        profits = []
        for backtest in self.backtests:
            profit = backtest.data.profit
            profit.name = backtest.params_text
            profits.append(profit)
        profits = pd.DataFrame(
            {backtest.params_text: backtest.data.profit for backtest in self.backtests}
        )
        profits.resample("1h").min().plot()

    def save(self):
        for backtest in self.backtests:
            backtest.save()

    def sort_by_profit_per_loss(self):
        return self.backtests.sort(key=lambda x: x.profit_per_loss, reversed=True)

    def sort_by_sharpe(self):
        return self.backtests.sort(key=lambda x: x.sharpe, reversed=True)
