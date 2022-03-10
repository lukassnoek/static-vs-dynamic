import sys
import logging
import numpy as np
import pandas as pd
import pymc3 as pm
import theano.tensor as tt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


# To go from str to int and back
ohe = OneHotEncoder(sparse=False)
ohe.fit(np.arange(6)[:, None])


def cross_val_predict_and_score(estimator, X, y, cv, scoring, classification=True):
    """ Cross-validation function which also keeps track of coefficients.

    X : DataFrame
        Pandas df with predictors
    y : Series
        Pandas series with target
    cv : cv object
        Sklearn cross-validator object
    scoring : func
        Scoring function
    classification : bool
        Whether it is a classification analysis (True) or a
        regression analysis (False)
    """
    
    if classification:
        K = y.unique().size
        classes = np.sort(y.unique())
    else:
        K = 1  # hacky
        classes = ['target']

    N = y.shape[0]
    n_reps = 1 if not hasattr(cv, 'n_repeats') else cv.n_repeats
    n_splits = cv.get_n_splits() // n_reps
    cv_gen = cv.split(X, y)
    
    # Pre-allocate predictions
    preds = [pd.DataFrame(np.zeros((N, K)), columns=classes, index=X.index)
             for _ in range(n_reps)]
    
    estimators = []  # to save for later
    i_rep = 0
    sigma = []
    for i, (train_idx_int, test_idx_int) in enumerate(cv_gen):

        train_idx = y.index[train_idx_int]
        test_idx = y.index[test_idx_int]
        
        X_train = X.loc[train_idx, :]
        y_train = y.loc[train_idx]
        estimator.fit(X_train, y_train)
        
        X_test = X.loc[test_idx, :]
        if classification:
            preds[i_rep].loc[test_idx, :] = estimator.predict_proba(X_test)
        else:
            y_pred = estimator.predict(X_test)[:, None]
            preds[i_rep].loc[test_idx] = y_pred
            y_test = y.loc[test_idx]
            # estimate sigma
            this_s = np.sum((y_test - y_pred.squeeze()) ** 2) / (X_test.shape[0] - X.shape[1])
            sigma.append(this_s)
        
        estimators.append(estimator)

        if (i + 1) % n_splits == 0:
            i_rep += 1

    scores = np.zeros((n_reps, K))
    for i in range(n_reps):
        if classification:
            these_preds = preds[i].values
        else:
            these_preds = preds[i].values

        if classification:
            y_ohe = ohe.transform(y.values[:, np.newaxis])
            scores[i, :] = scoring(y_ohe, these_preds, average=None)
        else:
            scores[i] = scoring(y.values, these_preds.squeeze())

    # Average across
    scores = scores.mean(axis=0)

    for i in range(n_reps):
        preds[i]['y_true'] = y.values

    # Get mean coefs (n folds x classes x features)
    coef = []
    for i, est in enumerate(estimators):  # loop over folds
        if hasattr(est, 'best_estimator_'):  # is GridSearchCV
            est = est.best_estimator_

        # Store coefficients
        if hasattr(est, 'coef_'):
            if classification:
                coef.append(np.c_[est.intercept_, est.coef_])
            else:
                coef.append(np.r_[est.intercept_, sigma[i], est.coef_][:, None])

    coef = np.stack(coef).squeeze()
    coef = np.mean(coef, axis=0)  # average coefs across folds
    return preds, scores, coef, est


def run_inverted_logreg(X, beta_hat, alpha_hat, standardize=True, draws=10_000, return_hdp=True):

    if standardize:
        X = X / X.std(axis=0)
        
    X_min, X_max = X.min(axis=0), X.max(axis=0)
    
    P = X.shape[1]
    K = alpha_hat.size
    with pm.Model() as model:

        X_ = tt.stack([pm.Uniform(f'X{i}_', lower=X_min[i], upper=X_max[i], shape=K)
                       for i in range(P)], axis=1)
        mu_ = tt.dot(X_, beta_hat.T) + alpha_hat
        py_ = pm.Deterministic('py_', tt.nnet.softmax(mu_))
        y_ = pm.Categorical('y_', p=py_, observed=range(K))

        trace = pm.sample(draws=draws)
        X_hdp = np.zeros((K, P))
        for i in range(K):
            for ii in range(P):
                X_hdp[i, ii] = pm.stats.hdi(trace[f'X{ii}_'][:, i], hdi_prob=0.05).mean()
        
        if return_hdp:
            return trace, X_hdp
        else:
            return trace


def run_inverted_linreg(X, beta_hat, alpha_hat, sigma_hat, y=[-.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6],
                        standardize=True, draws=10_000, return_hdp=True):
    if standardize:
        X = X / X.std(axis=0)

    this_y = np.array(y) - alpha_hat
    print(f"Actual y: {this_y}")

    X_min, X_max = X.min(axis=0), X.max(axis=0)
    P = X.shape[1]
    K = len(y)
    with pm.Model() as model:

        X_ = tt.stack([pm.Uniform(f'X{i}_', lower=X_min[i], upper=X_max[i], shape=K)
                       for i in range(P)], axis=1)
        mu_ = tt.dot(X_, beta_hat)
        y_ = pm.Normal("y_", mu=mu_, sigma=sigma_hat, observed=this_y)
        trace = pm.sample(draws=draws)
        
        X_hdp = np.zeros((K, P))
        for i in range(K):
            for ii in range(P):
                X_hdp[i, ii] = pm.stats.hdi(trace[f'X{ii}_'][:, i], hdi_prob=0.05).mean()
        
        if return_hdp:
            return trace, X_hdp
        else:
            return trace
