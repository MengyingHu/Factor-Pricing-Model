import numpy as np
import os
import pandas as pd
from cached_property import cached_property

class Statistics():
    """
    compute some statistics information for factors
    """

    def __init__(self, var_data, market_return_data, risk_free_data, df, sort_column,**kwargs):
        """
        Args:

        """
        self.var = var_data
        self.market_return = market_return_data
        self.rf = risk_free_data
        self.df = df
        self.sort_column = sort_column

    @classmethod
    def create(cls, path_datafile, sheet_var, sheet_mr, sheet_rf, sheet_mv, sheet_bm, sheet_stocks, sort_column, **kwargs):
        # import data
        # data is time by ISIN
        # ToDo: delete transform
        var_data = pd.read_excel(path_datafile, sheet_name=sheet_var, index_col=0)
        # dataframe, time by index
        market_return_data = pd.read_excel(path_datafile, sheet_name=sheet_mr, index_col=0)
        # dataframe, time by type
        risk_free_data = pd.read_excel(path_datafile, sheet_name=sheet_rf, index_col=0)
        market_value_data = pd.read_excel(path_datafile, sheet_name=sheet_mv, index_col=0)
        book_to_market_data = pd.read_excel(path_datafile, sheet_name=sheet_bm, index_col=0)
        stocks_total_return_index_data = pd.read_excel(path_datafile, sheet_name=sheet_stocks, index_col=0)
        total_return = stocks_total_return_index_data[1:] / stocks_total_return_index_data[:-1].values - 1
        frame = [total_return, market_value_data, book_to_market_data]
        df = pd.concat(frame, keys=['total return', 'market value', 'book to market'])

        df = df.T.loc[sort_column.index]
        df.dropna(axis=1, how='any', inplace=True)
        # df is dataframe, ISIN by time
        return cls(var=var_data, market_return=market_return_data, rf=risk_free_data, df=df, sort_column=sort_column)

    def Mkt(self):
        '''
        :return: Mkt is dataframe, excess return by time
        '''
        Mkt = pd.DataFrame([self.market_return - self.rf], index=self.rf.index)
        return Mkt.T

   ''' def beta(self, period='5A'):
        beta = pd.
        return beta'''


    def SMB(self):
        '''
        :return: SMB is dataframe, smb by time
        '''
        n = np.ceil(self.df.index.shape / 2)

        import ipbd; ipbd.set_trace()

        # ToDo: check the expression of self.df[col].sort['market value'][:n].index['total return']
        SMB = pd.DataFrame([self.df[col].sort['market value'][:n].index['total return'].mean
                            - self.df[col].sort['MV'][n:].index['total return'].mean
                           for col in self.df.columns], columns=self.df.columns)
        return SMB

    def HML(self):
        '''
        :return: HML is dataframe, hml by time
        '''
        n = np.ceil(0.3 * self.df.index.shape)
        m = np.ceil(0.7 * self.df.index.shape)
        # ToDo: check self.df[col].sort['book to market'][:n].index['total return'].mean
        HML = pd.DataFrame([self.df[col].sort['book to market'][:n].index['total return'].mean
                            - self.df[col].sort['book to market'][m:].index['total return'].mean
                           for col in self.df.columns], columns=self.df.columns)
        return HML

    def var(self):
        '''
        :return: var is dataframe, var return by time
        '''
        n = np.ceil(0.2 * self.df.index.shape)
        m = np.ceil(0.8 * self.df.index.shape)
        self.df[self.sort_column] = self.var[self.sort_column]
        self.df.sort(['sort_column'], inplace=True)
        # ToDo: check self.df[:n].index['total return']
        var = pd.DataFrame([self.df[:n].index['total return'] - self.df[m:].index['total return']], columns=self.df.columns)
        return var

    def information(self, factor):
        '''
        :param factor: one of Mkt, SMB, HML var
        '''
        mean = factor.mean(axis=1)
        std = factor.std(axis=1)
        information = pd.DataFrame({'mean':mean, 'std':std}, index=str(factor))
        return information

    def































