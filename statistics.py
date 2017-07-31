import numpy as np
import os
import pandas as pd

import statsmodels.api as sm

from cached_property import cached_property

class Statistics():
    """
    compute some statistics information for factors
    """

    def __init__(self, var_data, market_return_data, risk_free_data, total_return_data, market_value_data,
                 book_to_market_data, sort_column, **kwargs):
        """
        Args:

        """
        self.var = var_data
        self.market_return = market_return_data
        self.rf = risk_free_data
        self.total_return = total_return_data
        self.market_value = market_value_data
        self.book_to_market = book_to_market_data
        self.sort_column = sort_column


    @classmethod
    def create(cls, path_data, sheet_var, sheet_mr, sheet_rf, sheet_mv, sheet_bm, sheet_stocks, sort_column, **kwargs):
        # import data
        # data is time by ISIN
        # ToDo: pay attention to columns and rows
        # pay attention to difference between read_csv and read_excel, for read_csv, the data format should be same
        var_data = pd.read_csv(os.path.join(path_data, sheet_var + ".csv"), delimiter=';', index_col=0)
        # dataframe, time by index
        market_return_data = pd.read_csv(os.path.join(path_data, sheet_mr + ".csv"), delimiter=';', index_col=0)
        # dataframe, time by type
        risk_free_data = pd.read_csv(os.path.join(path_data, sheet_rf + ".csv"), delimiter=';', index_col=0)
        market_value_data = pd.read_csv(os.path.join(path_data, sheet_mv + ".csv"), delimiter=';', index_col=0)
        book_to_market_data = pd.read_csv(os.path.join(path_data, sheet_bm + ".csv"), delimiter=';', index_col=0)
        stocks_total_return_index_data = pd.read_csv(os.path.join(path_data, sheet_stocks + ".csv"), delimiter=';', index_col=0)

        # compute periodical monthly data
        market_value = (market_value_data[:-1] + market_value_data[1:].values) / 2
        book_to_market = (book_to_market_data[:-1] + book_to_market_data[1:].values) / 2
        market_return = market_return_data[1:] / market_return_data[:-1].values - 1
        market_return.index = market_value.index
        total_return = stocks_total_return_index_data[1:] / stocks_total_return_index_data[:-1].values - 1
        total_return.index = market_value.index

        risk_free_data = risk_free_data.T['US DOLLAR 3M DEPOSIT (FT/TR) - MIDDLE RATE'][:-1]
        risk_free = pd.DataFrame({'3M DEPOSIT': risk_free_data.values / 100}, index=market_value.index)

        # form multiindex dataframe for total return, size, book to market
        #frame = [total_return, market_value, book_to_market]
        #df = pd.concat(frame, keys=['total return', 'market value', 'book to market'])

        total_return.dropna(axis=1, how='any', inplace=True)
        market_value.dropna(axis=1, how='any', inplace=True)
        book_to_market.dropna(axis=1, how='any', inplace=True)
        var_data.dropna(axis=0, how='any', inplace=True)
        # df is dataframe, ISIN by time
        return cls(var_data=var_data, market_return_data=market_return, risk_free_data=risk_free,
                   market_value_data=market_value, book_to_market_data=book_to_market,
                   total_return_data=total_return, sort_column=sort_column)

    @cached_property
    def Mkt(self):
        '''
        :return: Mkt is dataframe, time by excess return
        '''
        # in dataframe subtract, the latter dataframe should be values
        return (self.market_return - self.rf.values)

    def beta(self):
       '''

       :return: index by beta
       '''
       import ipdb;ipdb.set_trace()
       # ToDo: check self.Mkt
       beta = pd.DataFrame([sm.OLS( self.total_return[col].values, self.Mkt.values).fit().params
                            for col in self.total_return.columns], index=self.total_return.columns)
       return beta


    def SMB(self):
        '''
        :return: SMB is dataframe, time by smb
        '''
        m = int(np.ceil(self.market_value.columns.shape[0] / 2))
        SMB = pd.DataFrame([pd.DataFrame([self.total_return.loc[index], self.market_value.loc[index]],
                        index=['total return', 'market value']).sort_values('market value', axis=1).values[0,:m].mean()
                        - pd.DataFrame([self.total_return.loc[index], self.market_value.loc[index]],
                        index=['total return', 'market value']).sort_values('market value', axis=1).values[0,m:].mean()
                        for index in self.total_return.index], index = self.total_return.index)
        # ToDo: check the expression of self.df[col].sort['market value'][:n].index['total return']
        #SMB = pd.DataFrame([self.df[index].sort['market value'][:n].index['total return'].mean
        #                   - self.df[index].sort['MV'][n:].index['total return'].mean
        #                  for index in self.df.index], index=self.df.index)
        return SMB

    def HML(self):
        '''
        :return: HML is dataframe, hml by time
        '''
        n = int(np.ceil(0.3 * self.total_return.columns.shape[0]))
        m = int(np.ceil(0.7 * self.total_return.columns.shape[0]))
        # ToDo: check self.df[col].sort['book to market'][:n].index['total return'].mean

        HML = pd.DataFrame([pd.DataFrame([self.total_return.loc[index], self.book_to_market.loc[index]],
                        index=['total return', 'book to market']).sort_values('book to market', axis=1).values[0,:n].mean()
                        - pd.DataFrame([self.total_return.loc[index], self.book_to_market.loc[index]],
                        index=['total return', 'book to market']).sort_values('book to market', axis=1).values[0,m:].mean()
                        for index in self.total_return.index], index = self.total_return.index)
        return HML

    def VAR(self):
        '''
        :return: var is dataframe, var return by time
        '''
        n = int(np.ceil(0.2 * self.total_return.columns.shape[0]))
        m = int(np.ceil(0.8 * self.total_return.columns.shape[0]))
        self.total_return.loc[self.sort_column] = self.var[self.sort_column]

        self.total_return.sort_values([self.sort_column], axis=1, inplace=True)
        # ToDo: delete sort_column
        # ToDo: check self.df[:n].index['total return']
        del self.total_return.loc[self.sort_column]
        VAR = pd.DataFrame([self.total_return.loc[index].values[:n].mean() - self.total_return.loc[index].values[m:].mean()
                        for index in self.total_return.index], index = self.total_return.index)
        return VAR

    def information(self, factor):
        '''
        :param factor: one of Mkt, SMB, HML var
        '''
        mean = factor.mean()
        std = factor.std()
        information = pd.DataFrame([mean, std], index=['mean', 'std'])
        return information

    def two_variables_metric(self,variable_1, variable_2, k=5):
        '''
        :param:
        variables can be beta, total_return, market_value, book_to_market, var
        :return: table to output return sorted by variables
        '''
        # ToDo: check self.variables
        variable_one = self.str(variable_1)
        variable_two = self.str(variable_2)
        df = self.market_return.T
        import ipdb;ipdb.set_trace()
        if (variable_one.index == variable_two.index and
            variable_one.columns == variable_two.columns):

            df[variable_1, variable_2] = [variable_one,variable_two]
            df = df.sort_values(variable_1)

            group_size_one = np.ceil(len(df) / k)
            grouped_one = df.groupby(np.arange(len(df)) // group_size_one).sort_values(variable_2)

            group_size_two = np.ceil(len(grouped_one) / k)
            two_variables_metric = df.groupby(np.arange(len(grouped_one)) // group_size_two).mean()

        else:
            print 'format error of variables'
        return two_variables_metric
