import pandas as pd

info = pd.read_csv('../FEED/stims/stimuli-expressive_selection-all.csv', sep='\t', index_col=0)
info = info.loc[:, info.columns.str.contains('face_1')].sort_index()
info.to_csv('data/features/faceID.tsv', sep='\t')