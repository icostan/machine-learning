from utils import log
import pandas as pd


def extract(X, feature, extractor):
    names = extractor.get_feature_names()
    xt = extractor.transform(X[feature].values).toarray()
    xd = pd.DataFrame(xt, columns=names, index=X.index)
    log('E ' + feature, xd.shape)
    return xd
