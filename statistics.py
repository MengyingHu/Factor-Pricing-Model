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
        #market_value = (market_value_data[:-1] + market_value_data[1:].values) / 2
        market_value = market_value_data[:-1]
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
        var_data.dropna(axis=0, how='any', inplace=True)
        total_return = total_return[var_data.index]
        total_return.dropna(axis=1, how='any', inplace=True)
        market_value = market_value[total_return.columns].fillna(method='bfill')
        book_to_market = book_to_market[total_return.columns].fillna(method='bfill')


        # df is dataframe, ISIN by time
        return cls(var_data=var_data, market_return_data=market_return, risk_free_data=risk_free,
                   market_value_data=market_value, book_to_market_data=book_to_market,
                   total_return_data=total_return, sort_column=sort_column)

    @cached_property
    def market_excess_return(self):
        '''
        :return: Mkt is dataframe, time by excess return
        '''
        # in dataframe subtract, the latter dataframe should be values
        return (self.market_return - self.rf.values)

    # cached_property let beta can be self.beta in portfolio class, but beta can not be called out of portfolio class

    @cached_property
    def firm_excess_return(self):
        return pd.DataFrame(self.total_return.values - self.rf.values, columns=self.total_return.columns)

    def beta(self):
       '''

       :return: index by beta
       '''
       # ToDo: check self.Mkt
       X = sm.add_constant(self.market_excess_return.values)
       beta = pd.DataFrame([sm.OLS(self.firm_excess_return[col].values, X).fit().params[1] for col in self.total_return.columns], index=self.total_return.columns)
       return beta

    @cached_property
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

    @cached_property
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

    @cached_property
    def VAR(self):
        '''
        :return: var is dataframe, var return by time
        '''
        n = int(np.ceil(0.25 * self.total_return.columns.shape[0]))
        m = int(np.ceil(0.75 * self.total_return.columns.shape[0]))
        self.total_return.loc[self.sort_column] = self.var[self.sort_column]

        self.total_return.sort_values([self.sort_column], axis=1, inplace=True)
        # ToDo: delete sort_column
        # ToDo: check self.df[:n].index['total return']
        self.total_return = self.total_return.drop(self.total_return.index[-1])
        VAR = pd.DataFrame([self.total_return.loc[index].values[:n].mean() - self.total_return.loc[index].values[m:].mean()
                        for index in self.total_return.index], index= self.total_return.index)
        return VAR

    def information(self, factor):
        '''
        :param factor: one of 'Mkt', 'SMB', 'HML', 'VAR'
        '''
        if factor == 'SMB':
            factor = self.SMB
        elif factor == 'HML':
            factor = self.HML
        elif factor == 'VAR':
            factor = self.VAR
        elif factor == 'Mkt':
            factor = self.excess_return

        mean = factor.mean()
        std = factor.std()
        min = factor.min()
        max = factor.max()
        # TODO: T-stat
        information = pd.DataFrame([mean, std, min, max], index=['mean', 'std', 'min', 'max'])
        return information

    def correlation(self):
        import ipdb;ipdb.set_trace()
        frame = [self.excess_return, self.SMB, self.HML, self.VAR]
        df = pd.concat(frame,axis=1)
        corr_matrix = df.corr()
        return corr_matrix

    def two_variables_metric(self,variable_1, variable_2, k=5):
        '''
        :param:
        variable_1 can be time variation, total_return, market_value, book_to_market
        vairable_2 can be time constant, var, beta
        :return: table to output return sorted by variables
        '''
        # ToDo: check self.variables

        df = self.total_return.T
        # ToDo: check  if variable_1.columns ==variable_2.columns
        # time constant variable sorted
        df[str(variable_2)] = variable_2.T

        df = df.sort_values(str(variable_2))
        del df[str(variable_2)]
        group_size_2 = len(df) / float(k)
        grouped_2 = df.groupby(np.ceil((np.arange(len(df)) + 1) / group_size_2) - 1)

        two_variables_metric = pd.DataFrame({}, index=['beta_low', 'beta_2', 'beta_3', 'beta_4', 'beta_high'],
                               columns=range(k))
        for name, group in grouped_2:
            group_daily = pd.DataFrame({}, index=['beta_low', 'beta_2', 'beta_3', 'beta_4', 'beta_high'])
            for date in group.columns:
                grouped_1 = pd.DataFrame(dict(s1=group[date], s2=variable_1.T[date]), index=group[date].index)
                sorted_1 = grouped_1.sort_values(grouped_1.columns[1])
                size = int(len(sorted_1) / float(k))
                group_daily[date] = pd.DataFrame([grouped_1[grouped_1.columns[0]].values[size * m:size * (m+1)].mean() for m in range(k)], index=group_daily.index, columns=['beta'])['beta']
                group_daily[date].loc['beta_high'] = pd.DataFrame([grouped_1[grouped_1.columns[0]].values[size * (k-2):].mean()], columns=['beta'])['beta'].values[0]
            two_variables_metric[name] = group_daily.mean(axis=1)

        # time variation variable sorted
        return two_variables_metric.mean(axis=1)

    def firm_specific_regressions(self):
        '''

        :return: average coefficients of factor regressions

        table 2 of 'IS ACCRUALS QUALITY A PRICED RISK FACTOR'
        '''
        frame = [self.market_excess_return, self.SMB, self.HML, self.VAR]
        X = pd.concat(frame, axis=1).values
        X = sm.add_constant(X)
        params = pd.DataFrame([sm.OLS(self.firm_excess_return[col].values, X).fit().params for col in self.total_return.columns], index=self.total_return.columns)
        t_stat = pd.DataFrame([sm.OLS(self.firm_excess_return[col].values, X).fit().tvalues for col in self.total_return.columns], index=self.total_return.columns)
        return params.mean(axis=0), t_stat.mean(axis=0)

    def Fama_MacBeth(self):



