
import os
import pandas as pd

import sys
sys.path.append("/home/hum/Python_projects/Backtesting/carbon-delta-performance/Factor-Pricing-Model")
# Now, this should work:

from module.Statistics_Information import Statistics
from settings import path_data, filename

def main():

    path_datafile = os.path.join(path_data, filename + ".xlsx")

    portfolio = Statistics.creat(path_datafile=path_datafile, sheet_var='VaR', sheet_mr='SP500', sheet_rf='rf',
                                 sheet_mv='MV', sheet_bm='PE', sheet_stocks='total return', sort_column='VAR REGULATIONS [%]'):

if __name__ == "__main__":
    main()