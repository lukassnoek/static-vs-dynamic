import numpy as np
import pandas as pd
from glob import glob


tsvs = sorted(glob('data/ratings/*expressive*.tsv'))
for tsv in tsvs:
    df = pd.read_csv(tsv, sep='\t', index_col=0)
    df.loc[df['data_split'] == 'test_extra', 'data_split'] = 'train'
    #tmp = df.query("data_split == 'test' & rating_type == 'emotion'")
    #tmp['face_id'] = [idx.split('_')[0] for idx in tmp.index]
    #exit()
    s1_repeats = df.query("session == 1 & data_split != 'test'").index.unique()
    idx = np.random.choice(s1_repeats.tolist(), replace=False, size=94)
    df.loc[idx, 'data_split'] = 'test_extra'
    df.to_csv(tsv, sep='\t')