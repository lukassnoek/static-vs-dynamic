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


def run_subjects_parallel(sub, target, feature_spaces, model, cv, n_comp):
    """ Helper function to parallelize analysis across subjects. """
    
    classification = True if target == 'emotion' else False
    scoring = tjur_score if classification else r2_score

    scores, preds, coefs = [], [], dict()
    for fs in feature_spaces:
        
        if not isinstance(fs, (tuple, list)):
            fs = (fs,)
        
        fs_name = ''.join(fs)

        # Load data
        dl = DataLoader(sub=sub, log_level=30)
        dl.load_y(target=target, data_split='train')
        dl.load_X(feature_set=fs, n_comp=n_comp, standardize=True)
        X, y = dl.return_Xy()
        
        preds_, scores_, coefs_, _ = cross_val_predict_and_score(
            estimator=model,
            X=X, y=y,
            cv=cv,
            classification=classification,
            scoring=scoring,
        )

        dl.log.warning(f"sub-{sub} scores: {scores_.round(2)} (fs = {fs_name})")
        scores_df = pd.DataFrame(scores_, columns=['score'])
        
        if classification:
            scores_df['emotion'] = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
                
        scores_df['sub'] = sub
        scores_df['feature_set'] = fs_name
        scores.append(scores_df)

        for i in range(len(preds_)):
            preds_[i]['feature_set'] = fs_name
            preds_[i]['sub'] = sub
            preds_[i]['rep'] = i
        
        preds.append(pd.concat(preds_, axis=0))
        if classification:
            coefs_df = pd.DataFrame(data=coefs_, columns=['icept'] + X.columns.tolist())
            coefs_df['emotion'] = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
        else:
            coefs_df = pd.DataFrame(data=[coefs_], columns=['icept', 'sigma'] + X.columns.tolist())
        
        coefs_df['sub'] = sub
        coefs_df['feature_set'] = fs_name
        coefs[fs_name] = coefs_df

    scores = pd.concat(scores, axis=0)
    preds = pd.concat(preds, axis=0)
    return preds, scores, coefs


if __name__ == '__main__':
    
    SUBS = [str(s).zfill(2) for s in range(1, 14)]
    FEATURE_SPACES = [
        'vertexPCA_type-static',
        #'vertexPCA_type-dynamic',
        #['vertexPCA_type-static', '+', 'vertexPCA_type-dynamic']
    ]
    TARGET = 'emotion'
    N_SPLITS = 10
    if TARGET in ['valence', 'arousal']:
        classification = False
        model = Ridge(fit_intercept=True, alpha=500)
        cv = RepeatedKFold(n_repeats=10, n_splits=N_SPLITS)
    else:
        classifiction = True
        cv = RepeatedStratifiedKFold(n_repeats=10, n_splits=N_SPLITS)
        model = LogisticRegression(
            class_weight='balanced', n_jobs=1, C=10, max_iter=1000,
            fit_intercept=True, solver='liblinear'
        )

    # RUN ANALYSIS IN PARALLEL
    out = Parallel(n_jobs=13)(delayed(run_subjects_parallel)
        (sub, TARGET, FEATURE_SPACES, model, cv, 50) for sub in SUBS
    )

    # Concatenate predictions and scores across participants
    preds = pd.concat([o[0] for o in out], axis=0)
    scores = pd.concat([o[1] for o in out], axis=0)
    print(scores.groupby('feature_set').mean())

    coefs = [o[2] for o in out]

    # Save
    root_dir = op.dirname(op.dirname(op.dirname(__file__)))
    f_out = op.join(root_dir, 'results', 'optimization', f'target-{TARGET}_scores.tsv')
    scores.to_csv(f_out, sep='\t')
    preds.to_csv(f_out.replace('_scores', '_preds'), sep='\t')

    for fs in FEATURE_SPACES:
        
        if not isinstance(fs, (tuple, list)):
            fs = (fs,)

        fs_name = ''.join(fs)
        coefs_df = pd.concat([o[fs_name] for o in coefs], axis=0)    
        f_out = op.join(root_dir, 'results', 'optimization', f'target-{TARGET}_fs-{fs_name}_coefs.tsv')
        coefs_df.to_csv(f_out, sep='\t')