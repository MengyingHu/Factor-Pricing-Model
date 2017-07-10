import numpy as np
import os
import pandas as pd
from cached_property import cached_property

class Statistics():
    """
    compute some statistics information for factors
    """

    def __init__(self, var_data, market_return_data, risk_free_data, market_value_data, book_to_market_data, stocks_total_return_index_data, **kwargs):
        """
        Args:

        """
        self.var = var_data
        self.market_return = market_return_data
        self.rf = risk_free_data
        self.market_value = market_value_data
        self.book_to_market = book_to_market_data
        self.stocks_total_return = stocks_total_return_index_data

    @classmethod
    def create(cls, path_datafile, sheet_var, sheet_mr, sheet_rf, sheet_mv, sheet_bm, sheet_stocks,**kwargs):
        # import data
        var_data = pd.read_excel(path_datafile, sheet_name=sheet_var, index_col=0)
        market_return_data = pd.read_excel(path_datafile, sheet_name=sheet_mr, index_col=0)
        risk_free_data = pd.read_excel(path_datafile, sheet_name=sheet_rf, index_col=0)
        market_value_data = pd.read_excel(path_datafile, sheet_name=sheet_mv, index_col=0)
        book_to_market_data = pd.read_excel(path_datafile, sheet_name=sheet_bm, index_col=0)
        stocks_total_return_index_data = pd.read_excel(path_datafile, sheet_name=sheet_stocks, index_col=0)
        return cls(var=var_data, market_return=market_return_data, rf=risk_free_data, market_value=market_value_data,
                   book_to_market=book_to_market_data, stocks_total_return=stocks_total_return_index_data)

    def data_cleaning(self, sort_column):
        self.stocks_total_return[sort_column] = self.var[sort_column]
        self.stocks_total_return = self.stocks_total_return.sort_values(sort_column)
        self.stocks_total_return(axis=1, how='any', inplace=True)





























