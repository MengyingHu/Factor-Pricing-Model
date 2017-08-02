
import os
import pandas as pd

import sys
sys.path.append("/home/hum/Python_projects/Thesis_Factor_pricing_model/Factor-Pricing-Model")
# Now, this should work:

from statistics import Statistics
from settings import path_data, filename

def main():
    path = path_data

    portfolio = Statistics.create(path_data=path, sheet_var='VaR', sheet_mr='S&P500', sheet_rf='rf',
                                  sheet_mv='m_v', sheet_bm='MB', sheet_stocks='total return',
                                  sort_column='Overall Climate Change Resilience Level [%]')
    beta = portfolio.beta()
    info = portfolio.correlation()
    info = portfolio.two_variables_metric(portfolio.book_to_market, var)

if __name__ == "__main__":
    main()
