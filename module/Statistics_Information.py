import numpy as np
import os
import pandas as pd
from cached_property import cached_property

class Statistics():
    """
    compute some statistics information for factors
    """

    def __init__(self, var_data, market_return_data, risk_free_data, df, **kwargs):
        """
        Args:

        """
        self.var = var_data
        self.market_return = market_return_data
        self.rf = risk_free_data
        self.df = df

    @classmethod
    def create(cls, path_datafile, sheet_var, sheet_mr, sheet_rf, sheet_mv, sheet_bm, sheet_stocks, sort_column, **kwargs):
        # import data
        # data is time by ISIN
        # ToDo: delete transform
        var_data = pd.read_excel(path_datafile, sheet_name=sheet_var, index_col=0)
        # dataframe, time by index
        market_return_data = pd.read_excel(path_datafile, sheet_name=sheet_mr, index_col=0)
        # dataframe, time by type
        risk_free_data = pd.read_excel(path_datafile, sheet_name=sheet_rf, index_col=0).T
        market_value_data = pd.read_excel(path_datafile, sheet_name=sheet_mv, index_col=0).T
        book_to_market_data = pd.read_excel(path_datafile, sheet_name=sheet_bm, index_col=0).T
        stocks_total_return_index_data = pd.read_excel(path_datafile, sheet_name=sheet_stocks, index_col=0)

        frame = [stocks_total_return_index_data, market_value_data, book_to_market_data]
        df = pd.concat(frame, keys=['total return', 'market value', 'book to market'])

        df = df.T.loc[sort_column.index]
        df.dropna(axis=0, how='any', inplace=True)
        # df is dataframe, ISIN by time
        import ipdb; ipdb.set_trace()
        return cls(var=var_data, market_return=market_return_data, rf=risk_free_data, df=df)

    def SMB(self):



























