import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv('../FEED/stims/stimuli-expressive_selection-train+test.tsv', sep='\t', index_col=0)
df = df.sort_index()

au_cols = sorted([col for col in df.columns if col[:2] == 'AU'])
df = df.loc[:, au_cols] #+ ['data_split']]
df = df.drop('AU22', axis=1)  # issue with animation!
df['AU27i'] = df.loc[:, ['AU26', 'AU27i']].max(axis=1)  # same as AU26!
df = df.drop('AU26', axis=1)
df.to_csv('data/features/AU.tsv', sep='\t', index=True)