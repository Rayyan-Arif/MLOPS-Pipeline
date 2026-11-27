import numpy as np
np.Inf = np.inf

import pandas as pd
from deepchecks.tabular import Dataset, Suite
from deepchecks.tabular.checks import PercentOfNulls, DataDuplicates

df = pd.read_csv("data/processed/cleaned_retail.csv")
ds = Dataset(df, label=None)

suite = Suite(
    "CSV Validation",
    PercentOfNulls(columns=["Invoice", "StockCode"]).add_condition_percent_of_nulls_not_greater_than(0),
    DataDuplicates().add_condition_ratio_less_or_equal(0)
)

result = suite.run(ds)
deepchecks_passed = result.passed()

non_negative_ok = (df["Price"] >= 0).all()  
non_zero_ok     = (df["Quantity"] != 0).all()

if deepchecks_passed and non_negative_ok and non_zero_ok:
    print("Dataset passed all checks!")
else:
    print("Dataset failed validation!")
    result.show()