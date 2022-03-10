import os.path as op
import numpy as np
import pandas as pd
from glob import glob
from scipy.io import loadmat
from tqdm import tqdm
from sklearn.decomposition import PCA
from joblib import Parallel, delayed


N_COMPS = 50  # only extract 50 components
info = pd.read_csv('../FEED/stims/stimuli-expressive_selection-train+test.tsv', sep='\t', index_col=0)

# Find stimulus directories
dirs = sorted(glob('../FEED/FEED_stimulus_frames/id*'))
ids = np.array([op.basename(d).split('_')[0].split('-')[1] for d in dirs], dtype=str)
N = len(dirs)
index = [op.basename(d) for d in dirs]
ids = [f.split('_')[0] for f in index]
dm_ids = pd.get_dummies(ids).to_numpy()

def load_vertices(d):
    """ Load in all first and middle frame of a given stimulus. """
    vertices = np.zeros((2, 31049, 3))
    f01 = glob(f'{d}/*/texmap/frame01_vertices.mat')
    f15 = glob(f'{d}/*/texmap/frame15_vertices.mat')
    if len(f01) != 1 or len(f15) != 1:
        raise ValueError("Found >1 vertex files...")

    vertices[0, :, :] = loadmat(f01[0])['vertices']
    vertices[1, :, :] = loadmat(f15[0])['vertices']    
    return vertices

if op.isfile('data/vertices.npz'):
    print("Loading in previously aggregated vertex array ...")
    vertices = np.load('data/vertices.npz')['v']
else:
    out = Parallel(n_jobs=30)(delayed(load_vertices)(d) for d in tqdm(dirs))
    vertices = np.stack(out)
    np.savez_compressed('data/vertices.npz', v=vertices)

vs = []  # average across stims because frame01 is not the same for each stim
# from a given face
for id in tqdm(sorted(np.unique(ids))):
    idx = np.array([True if this_id == id else False for this_id in ids])
    av = vertices[idx, 0, :, :].mean(axis=0)
    vs.append(av)

vs = np.stack(vs)
vd = vertices[:, 1, :, :] - vertices[:, 0, :, :]

#### PCA dim reduction ####
colnames = [f'comp_{str(i).zfill(3)}' for i in range(N_COMPS)]
pca = PCA(n_components=N_COMPS)

### Static features
print("Fitting PCA on static features ...")
pca.fit(vs.reshape((50, 31049 * 3)))
np.savez('results/pca/pca_type-static_weights.npz', mu=pca.mean_, W=pca.components_)
vs_r = pca.transform(vs.reshape((50, 31049 * 3)))
df_f01 = pd.DataFrame(vs_r, columns=colnames, index=sorted(np.unique(ids)))
df_f01 = df_f01.loc[ids, :]  # tile!
df_f01.index = index  # use original index
df_f01.to_csv(f'data/features/vertexPCA_type-static.tsv', sep='\t')

### Dynamic features
print("Fitting PCA on dynamic features ...")
pca.fit(vd.reshape((N, 31049 * 3)))
np.savez(f'results/pca/pca_type-dynamic_weights.npz', mu=pca.mean_, W=pca.components_)
vd_r = pca.transform(vd.reshape((N, 31049 * 3)))
df_f15min01 = pd.DataFrame(vd_r, columns=colnames, index=index)
df_f15min01.to_csv(f'data/features/vertexPCA_type-dynamic.tsv', sep='\t')
