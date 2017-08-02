
import os
import pandas as pd

# Now, this should work:
import sys
sys.path.append("/home/hum/Python_projects/Thesis_Factor_pricing_model/Factor-Pricing-Model")

from statistics import Statistics
from settings import path_data

def main():
    path = path_data

    portfolio = Statistics.create(path_data=path, sheet_var='VaR', sheet_mr='MSCI', sheet_rf='rf',
                                  sheet_mv='MV', sheet_bm='MB', sheet_stocks='RI',
                                  sort_column='Overall Climate Change Resilience Level [%]')
    beta = portfolio.beta()
    info = portfolio.correlation()
    info = portfolio.two_variables_metric(portfolio.book_to_market, var)
    import ipdb;ipdb.set_trace()

if __name__ == "__main__":
    main()
