import sys
import numpy as np
import os.path as op
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, recall_score

sys.path.append(op.abspath(op.dirname(op.dirname(__file__))))
from data_io import DataLoader
from noiseceiling import compute_nc_classification, compute_nc_regression
from utils import tjur_score


subs = [str(s).zfill(2) for s in range(1, 14)]
ncs = []

for fs in ['vertexPCA_type-static', 'vertexPCA_type-dynamic',
           ['vertexPCA_type-static', '+', 'vertexPCA_type-dynamic']]:
    y_all, X_all = [], []
    for i, sub in enumerate(subs):
        dl = DataLoader(sub=sub, log_level=30)
        dl.load_y(target='emotion', data_split='test')
        dl.load_X(feature_set=fs, standardize=True, reduce_repeats=False)
        X, y = dl.return_Xy()
        nc = compute_nc_classification(X, y, use_repeats_only=True, use_index=False, score_func=tjur_score)

        dl.log.warning(f"Ceiling sub-{sub}: {nc.values.round(3)}")
        nc['sub'] = sub
        nc['feature_space'] = fs if isinstance(fs, str) else ''.join(fs)
        ncs.append(nc)

        y_all.append(y)
        X_all.append(X)

    y_all = pd.concat(y_all, axis=0)
    X_all = pd.concat(X_all, axis=0)
    ncb = compute_nc_classification(X_all, y_all, use_repeats_only=True, use_index=False, score_func=tjur_score)
    ncb['sub'] = 'between'

nc = pd.concat(ncs, axis=0)
nc.columns = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'sub', 'feature_space']
nc.to_csv('results/target-emotion_noiseceilings.tsv', sep='\t', index=False)


ncs = []
for fs in ['vertexPCA_type-static', 'vertexPCA_type-dynamic',
           ['vertexPCA_type-static', '+', 'vertexPCA_type-dynamic']]:

    nc = pd.DataFrame(np.zeros((len(subs), 4)), columns=['valence', 'arousal', 'sub', 'feature_space'])
    for target in ['valence', 'arousal']:
        y_all, X_all = [], []

        for i, sub in enumerate(subs):
            dl = DataLoader(sub=sub, log_level=30)
            dl.load_y(target=target, data_split='test')
            dl.load_X(feature_set=fs, standardize=False, reduce_repeats=False)
            X, y = dl.return_Xy()
            nc_ = compute_nc_regression(X, y, use_repeats_only=True, use_index=False).iloc[0]
            dl.log.warning(f"Ceiling sub-{sub}: {nc_}")
            nc.loc[i, target] = nc_
            nc.loc[i, 'feature_space'] = fs if isinstance(fs, str) else ''.join(fs)
            nc.loc[i, 'sub'] = sub
            y_all.append(y)
            X_all.append(X)
            ncs.append(nc)
        y_all = pd.concat(y_all, axis=0)
        X_all = pd.concat(X_all, axis=0)
        ncb = compute_nc_regression(X_all, y_all, use_repeats_only=True, use_index=False)    

nc = pd.concat(ncs, axis=0)
print(nc)
nc.to_csv('results/target-valencearousal_noiseceilings.tsv', sep='\t', index=False)

