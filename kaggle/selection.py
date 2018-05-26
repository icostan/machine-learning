from utils import log
from sklearn import feature_selection as fs
import pandas as pd


def threshold_filter(x, feature_names):
    selector = fs.VarianceThreshold(threshold=0.8)
    xv = selector.fit_transform(x)
    log('Threshold', selected_features(selector, feature_names))
    log('X', xv.shape)
    return xv, selector


def percentile_filter(X, y, percentile=20):
    selector = fs.SelectPercentile(fs.chi2, percentile=percentile)
    selector.fit(X, y)
    # features = selected_features(selector, feature_names)
    # log('Percentile', len(features))
    # log('X', xt.shape)
    # return pd.DataFrame(xt, columns=features, index=X.index), selector
    return selector


def kbest_filter(x, y, feature_names, k=10):
    selector = fs.SelectKBest(fs.chi2, k=k)
    xp = selector.fit_transform(x, y)
    features = selected_features(selector, feature_names)
    # log('KBest', len(features))
    # log('X', xp.shape)
    return pd.DataFrame(xp, columns=features, index=x.index), selector


def selected_features(selector, feature_names):
    idx = selector.get_support(feature_names)
    return list(map(lambda i: feature_names[i], idx))


def select(X, selector, feature_names):
    xt = selector.transform(X)
    features = selected_features(selector, feature_names)
    log('S', xt.shape)
    return pd.DataFrame(xt, columns=features, index=X.index)
