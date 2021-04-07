import sys
import numpy as np
import pandas as pd
sys.path.append('src')
from utils import plot_face, get_parameters
from model import run_inverted_linreg, run_inverted_logreg


N = 848  # number of unique stimuli
n_v = 31049  # number of vertices

# Load vertices, both static and dynamic
v = np.load('data/vertices.npz')['v']
mean_face = v[:, 0, :, :].mean(axis=0)  # for plotting
tris = np.load('data/tris.npy') - 1  # triangles

for aff in ['valence', 'arousal', 'emotion']:
    for tpe in ['static', 'dynamic']:
        coef = pd.read_csv(f'results/validation/target-{aff}_fs-vertexPCA_type-{tpe}_coefs.tsv', sep='\t', index_col=0)
        v_pca = pd.read_csv(f'data/features/vertexPCA_type-{tpe}.tsv', sep='\t', index_col=0).to_numpy()

        with np.load(f'results/pca/pca_type-{tpe}_weights.npz') as data:
            # "vd" stands for "vertex difference"
            mu_, w_ = data['mu'], data['W']
            if tpe == 'static':
                w_ = w_[:49, :]

        if tpe == 'static':
            v_ = v[:, 0, :, :]
            v_pca = v_pca[:, :49]
        else:
            v_ = v[:, 1, :, :]
            
        for sub in ['average'] + coef['sub'].unique().tolist():
            print(f"Running model: {aff}, {tpe}, sub-{sub}")
            if aff == 'emotion':
                alpha_hat, beta_hat, Z = get_parameters(sub, coef)
            else:
                alpha_hat, beta_hat, sigma_hat, Z = get_parameters(sub, coef)

            if tpe == 'static':
                if aff == 'emotion':
                    beta_hat = beta_hat[:, :49]
                else:
                    beta_hat = beta_hat[:49]

            draws = 10_000 if sub == 'average' else 1000
            
            if aff == 'emotion':
                trace, X_hdp = run_inverted_logreg(v_pca, beta_hat, alpha_hat, return_hdp=True, draws=draws)
                row_titles = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']
            else:
                trace, X_hdp = run_inverted_linreg(v_pca, beta_hat, alpha_hat, sigma_hat, return_hdp=True, draws=draws)
                row_titles = ['-0.75', '-0.50', '-0.25', '0.0', '+0.25', '+0.50', '+0.75']
            
            if tpe == 'static':
                S_emo = ((X_hdp * v_pca.std(axis=0)) @ w_).reshape((X_hdp.shape[0], n_v, 3))
            else:
                S_emo = ((X_hdp * v_pca.std(axis=0)) @ w_ + mu_).reshape((X_hdp.shape[0], n_v, 3))

            overlay_emo = S_emo / v_.std(axis=0)
            overlay_emo[np.isnan(overlay_emo)] = 0
            fig = plot_face(mean_face + S_emo, tris, overlay=overlay_emo, cmax=6, cmin=-6, threshold=0.1,
                            col_titles=['X', 'Y', 'Z'], row_titles=row_titles)
            fig.write_image(f'figures/reconstructions/sub-{str(sub).zfill(2)}_target-{aff}_type-{tpe}_recon.png', scale=2)