import pandas as pd
from enum import Enum
from numpy import log, log2, log10, square, sqrt, power
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, OrdinalEncoder

# Imputation Aggregation functions
class ImputationAggregations(Enum):
    none   = lambda x : x
    zero   = lambda _ : 0
    NA     = lambda _ : "NA"
    sum    = lambda x : x.sum(skipna = True)
    mean   = lambda x : x.mean(skipna = True)
    median = lambda x : x.median(skipna = True)
    mode   = lambda x : x.value_counts().index.to_list()[0]

# Feature-Transformations 
class Transformations(Enum):
    none            = lambda x : x
    onehotencoding  = lambda x : pd.get_dummies(x)
    ordinalencoding = lambda x : pd.DataFrame(OrdinalEncoder().fit_transform(x), columns=x.columns)
    minmaxscaler    = lambda x : pd.DataFrame(MinMaxScaler().fit_transform(x), columns=x.columns)
    standardscaler  = lambda x : StandardScaler().fit_transform(x)
    maxabsscaler    = lambda x : MaxAbsScaler().fit_transform(x)
    robustscaler    = lambda x : RobustScaler().fit_transform(x)
    log             = lambda x : log(x)
    log2            = lambda x : log2(x)
    log10           = lambda x : log10(x)
    square          = lambda x : square(x)
    sqrt            = lambda x : sqrt(x)
    cube            = lambda x : power(x,3)

# Imputation Aggregation functions mapping
imp_agg = {
    "none"   : ImputationAggregations.none,
    "zero"   : ImputationAggregations.zero,
    "NA"     : ImputationAggregations.NA,
    "sum"    : ImputationAggregations.sum,
    "mean"   : ImputationAggregations.mean,
    "median" : ImputationAggregations.median,
    "mode"   : ImputationAggregations.mode
}

# Feature-Transformations functions mapping
transformations = {
    "none"             : Transformations.none,
    "one-hot-encoding" : Transformations.onehotencoding,
    "ordinal-encoding" : Transformations.ordinalencoding,
    "min-max-scaler"   : Transformations.minmaxscaler,
    "standard-scaler"  : Transformations.standardscaler,
    "max-abs-scaler"   : Transformations.maxabsscaler,
    "robust-scaler"    : Transformations.robustscaler,
    "log"              : Transformations.log,
    "log2"             : Transformations.log2,
    "log10"            : Transformations.log10,
    "square"           : Transformations.square,
    "sqrt"             : Transformations.sqrt,
    "cube"             : Transformations.cube
}