import sys
import os.path as op
import numpy as np
import pandas as pd
from tqdm import tqdm
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

EMOTIONS = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']


for target in ['emotion', 'valence', 'arousal']:
    if target in ['valence', 'arousal']:
        classification = False
        model = Ridge(fit_intercept=False, alpha=500)
    else:
        classification = True
        model = LogisticRegression(
            class_weight='balanced', n_jobs=1, C=10, max_iter=1000,
            fit_intercept=True, solver='liblinear'
        )

    scores = []
    for i, sub in enumerate(SUBS):
        for fs in FEATURE_SPACES:
            n_comps = 50
            if not isinstance(fs, (tuple, list)):
                fs = (fs,)
            
            fs_name = ''.join(fs)

            # Load data
            dl = DataLoader(sub=sub, log_level=30)
            dl.load_y(target=target, data_split='train')
            dl.load_X(feature_set=fs, n_comp=n_comps, standardize=True, reduce_repeats=True)
            X_train, y_train = dl.return_Xy()
            if target != 'emotion':
                y_train_mean = y_train.mean()
                y_train = y_train - y_train_mean
        
            dl = DataLoader(sub=sub, log_level=30)
            dl.load_y(target=target, data_split='test')
            dl.load_X(feature_set=fs, n_comp=n_comps, standardize=True, reduce_repeats=False)
            X_test, y_test = dl.return_Xy()
            if target != 'emotion':
                y_test = y_test - y_test.mean()

            model.fit(X_train, y_train)
            for i in tqdm(range(1000), desc=f'sub-{sub}, fs: {fs_name}'):  # permutations
                y_test = y_test.sample(frac=1, replace=False)
                y_test.index = X_test.index

                if classification:
                    preds = pd.DataFrame(model.predict_proba(X_test), index=X_test.index, columns=EMOTIONS)
                    y_ohe = pd.get_dummies(y_test)                
                    scores_ = tjur_score(y_ohe.to_numpy(), preds.to_numpy(), average=None)
                else:
                    preds = pd.DataFrame(model.predict(X_test), index=X_test.index, columns=['y_pred'])
                    scores_ = np.array([r2_score(y_test, preds.to_numpy())])
       
                scores_df = pd.DataFrame(scores_, columns=['score'])
                scores_df['sub'] = sub
                scores_df['feature_set'] = fs_name
       
                if classification:
                    scores_df['emotion'] = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']

                scores.append(scores_df)
   
    # Save
    root_dir = op.dirname(op.dirname(op.dirname(__file__)))
    f_out = op.join(root_dir, 'results', 'validation', f'target-{target}_permutationscores.tsv')
    scores = pd.concat(scores, axis=0)
    print(scores.groupby(['sub', 'feature_set']).mean())
    scores.to_csv(f_out, sep='\t')
