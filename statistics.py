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
        var_data = pd.read_csv(os.path.join(path_data, sheet_var + ".csv"), delimiter=';', index_col=0)[sort_column]
        # dataframe, time by index
        market_return_data = pd.read_csv(os.path.join(path_data, sheet_mr + ".csv"), delimiter=';', index_col=0)
        # dataframe, time by type
        risk_free_data = pd.read_csv(os.path.join(path_data, sheet_rf + ".csv"), delimiter=';', index_col=0)
        market_value_data = pd.read_csv(os.path.join(path_data, sheet_mv + ".csv"), delimiter=';', index_col=0)
        book_to_market_data = pd.read_csv(os.path.join(path_data, sheet_bm + ".csv"), delimiter=';', index_col=0)
        stocks_total_return_index_data = pd.read_csv(os.path.join(path_data, sheet_stocks + ".csv"), delimiter=';', index_col=0)

        # convert 'Index' to 'PeriodIndex'
        market_value_data.index = pd.to_datetime(market_value_data.index)
        market_return_data.index = pd.to_datetime(market_return_data.index)
        stocks_total_return_index_data.index = pd.to_datetime(stocks_total_return_index_data.index)
        book_to_market_data.index = pd.to_datetime(book_to_market_data.index)

        # compute periodical monthly data
        #market_value = (market_value_data[:-1] + market_value_data[1:].values) / 2
        market_value = market_value_data[:-1]
        #book_to_market = (book_to_market_data[:-1] + book_to_market_data[1:].values) / 2
        book_to_market = book_to_market_data[:-1]
        market_return = market_return_data[1:] / market_return_data[:-1].values - 1
        total_return = stocks_total_return_index_data[1:] / stocks_total_return_index_data[:-1].values - 1
        total_return.index = market_value.index

        risk_free_data = risk_free_data.T['US DOLLAR 3M DEPOSIT (FT/TR) - MIDDLE RATE'][:-1]
        risk_free = pd.DataFrame({'3M DEPOSIT': risk_free_data.values / 100}, index=market_value.index)

        # form multiindex dataframe for total return, size, book to market
        #frame = [total_return, market_value, book_to_market]
        #df = pd.concat(frame, keys=['total return', 'market value', 'book to market'])
        var_data.dropna(axis=0, how='any', inplace=True)
        var_data = var_data.loc[total_return.columns]
        total_return.dropna(axis=1, how='any', inplace=True)
        var_data = var_data.loc[total_return.columns]
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

    def Mkt(self):
        '''
        :return: Mkt is dataframe, time by excess return
        '''
        # in dataframe subtract, the latter dataframe should be values
        return (self.market_return - self.rf.values)

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
        :return: HML is dataframe, time by hml
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
        :return: var is dataframe, time by var return
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
        mean = factor.mean()
        std = factor.std()
        min = factor.min()
        max = factor.max()
        # TODO: T-stat
        information = pd.DataFrame([mean, std, min, max], index=['mean', 'std', 'min', 'max'])
        return information

    def correlation(self, factor1, factor2, factor3, factor4):
        '''

        :param factors: frame of factors, like[Mkt, SMB, HML, VAR]
        :return:
        '''
        frame = [factor1, factor2, factor3, factor4]
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
        df = self.total_return
        # ToDo: check  if variable_1.columns ==variable_2.columns
        # time constant variable sorted
        df.loc['beta'] = variable_2.T.iloc[0]
        df = df.T.sort_values('beta')
        del df['beta']
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

    def firm_specific_regressions(self, factor1, factor2, factor3, factor4):
        '''
        :param: frame of factors, like[Mkt, SMB, HML, VAR]
        :return: average coefficients of factor regressions

        table 2 of 'IS ACCRUALS QUALITY A PRICED RISK FACTOR'
        '''
        frame = [factor1, factor2, factor3, factor4]
        X = pd.concat(frame, axis=1).values
        X = sm.add_constant(X)
        params = pd.DataFrame([sm.OLS(self.firm_excess_return[col].values, X).fit().params for col in self.total_return.columns], index=self.total_return.columns)
        t_stat = pd.DataFrame([sm.OLS(self.firm_excess_return[col].values, X).fit().tvalues for col in self.total_return.columns], index=self.total_return.columns)
        return params.mean(axis=0), t_stat.mean(axis=0)

    def table_3_size_BM(self, k=5, period_frequency='A'):
        '''

        :param variable_1:
        :param variable_2:
        :return:
        '''
        import ipdb;
        ipdb.set_trace()
        variable_1 = self.market_value.groupby(pd.TimeGrouper(freq=period_frequency))
        variable_2 = self.book_to_market.groupby(pd.TimeGrouper(freq=period_frequency))
        variable_sort = self.total_return.groupby(pd.TimeGrouper(freq=period_frequency))

        params = pd.DataFrame({}, columns=['Mkt', 'SMB', 'HML', 'VAR'])
        R_2 = pd.DataFrame({}, columns=['R_square'])
        for date in self.market_value.index:
            grouped_daily = pd.DataFrame(dict(s1=variable_sort.loc[date], s2=variable_1.loc[date], s3=variable_2.loc[date]))
            group_sorted_1 = grouped_daily.sort_values('s2')
            group_size_1 = len(group_sorted_1) / float(k)
            grouped_1 = group_sorted_1.groupby(np.ceil((np.arange(len(group_sorted_1)) + 1) / group_size_1) - 1)
            for name, group in grouped_1:
                import ipdb;
                ipdb.set_trace()
                group_sorted_2 = group.sort_values('s3')
                size = int(len(group_sorted_2) / float(k))



        return two_variables_metric.mean(axis=1)

    #def Fama_MacBeth(self):



