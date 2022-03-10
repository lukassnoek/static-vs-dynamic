import sys
import os.path as op
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import r2_score
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold

sys.path.append(op.abspath(op.dirname(op.dirname(__file__))))
from data_io import DataLoader
from model import cross_val_predict_and_score
from utils import tjur_score  # for classification

    
SUBS = [str(s).zfill(2) for s in range(1, 14)]
FEATURE_SPACES = [
    'vertexPCA_type-static',
    'vertexPCA_type-dynamic',
    ['vertexPCA_type-static', '+', 'vertexPCA_type-dynamic']
]
TARGET = 'valence'
EMOTIONS = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']

if TARGET in ['valence', 'arousal']:
    classification = False
    model = Ridge(fit_intercept=False, alpha=500)
else:
    classification = True
    model = LogisticRegression(
        class_weight='balanced', n_jobs=1, C=10, max_iter=1000,
        fit_intercept=True, solver='liblinear'
    )

scores = []
coefs = {}
y_preds = []

for i, sub in enumerate(SUBS):
    for fs in FEATURE_SPACES:
        n_comps = 50
        if not isinstance(fs, (tuple, list)):
            fs = (fs,)
        
        fs_name = ''.join(fs)
        if fs_name not in coefs:
            coefs[fs_name] = []

        # Load data
        dl = DataLoader(sub=sub, log_level=30)
        dl.load_y(target=TARGET, data_split='train')
        dl.load_X(feature_set=fs, n_comp=n_comps, standardize=True, reduce_repeats=True)
        X_train, y_train = dl.return_Xy()
        if TARGET != 'emotion':
            y_train_mean = y_train.mean()
            y_train = y_train - y_train_mean

        dl = DataLoader(sub=sub, log_level=30)
        dl.load_y(target=TARGET, data_split='test')
        dl.load_X(feature_set=fs, n_comp=n_comps, standardize=True, reduce_repeats=False)
        X_test, y_test = dl.return_Xy()
        if TARGET != 'emotion':
            y_test_mean = y_test.mean()
            y_test = y_test - y_test_mean
        
        model.fit(X_train, y_train)
    
        if classification:
            preds = pd.DataFrame(model.predict_proba(X_test), index=X_test.index, columns=EMOTIONS)
            coefs_ = np.c_[model.intercept_, model.coef_]
            coefs_df = pd.DataFrame(coefs_, columns=['icept'] + X_train.columns.tolist())
            coefs_df['emotion'] = EMOTIONS
            y_ohe = pd.get_dummies(y_test)
            
            scores_ = tjur_score(y_ohe.to_numpy(), preds.to_numpy(), average=None)
            preds.loc[:, 'y_true'] = y_test.copy()
        else:
            preds = pd.DataFrame(model.predict(X_test), index=X_test.index, columns=['y_pred'])
            N, K = X_test.shape 
            sigma = np.sqrt(np.sum((y_test - preds.to_numpy().squeeze()) ** 2) / (N - K))
            coefs_ = np.r_[y_train_mean, sigma, model.coef_][:, None]
            coefs_df = pd.DataFrame(coefs_.T, columns=['icept', 'sigma'] + X_train.columns.tolist())
            scores_ = np.array([r2_score(y_test, preds.to_numpy())])
            # This is done to make the plots more interpretable (doesn't change score)
            preds = preds + y_test_mean
            preds.loc[:, 'y_true'] = y_test + y_test_mean

        dl.log.warning(f"sub-{sub} scores: {scores_.round(2)} (fs = {fs_name})")
        scores_df = pd.DataFrame(scores_, columns=['score'])
        scores_df['sub'] = sub
        scores_df['feature_set'] = fs_name
        coefs_df['sub'] = sub
        coefs_df['feature_set'] = fs_name

        if classification:
            scores_df['emotion'] = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']

        coefs[fs_name].append(coefs_df)
        scores.append(scores_df)

        preds.loc[:, 'sub'] = sub
        preds.loc[:, 'feature_set'] = fs_name
        y_preds.append(preds)

# Save
root_dir = op.dirname(op.dirname(op.dirname(__file__)))
f_out = op.join(root_dir, 'results', 'validation', f'target-{TARGET}_scores.tsv')
scores = pd.concat(scores, axis=0)
print(scores.groupby('feature_set').mean())
scores.to_csv(f_out, sep='\t')

preds = pd.concat(y_preds, axis=0)
preds.to_csv(op.join(root_dir, 'results', 'validation', f'target-{TARGET}_preds.tsv'), sep='\t')

for fs in FEATURE_SPACES:
    
    if not isinstance(fs, (tuple, list)):
        fs = (fs,)

    fs_name = ''.join(fs)
    coefs_df = pd.concat(coefs[fs_name], axis=0)    
    f_out = op.join(root_dir, 'results', 'validation', f'target-{TARGET}_fs-{fs_name}_coefs.tsv')
    coefs_df.to_csv(f_out, sep='\t')