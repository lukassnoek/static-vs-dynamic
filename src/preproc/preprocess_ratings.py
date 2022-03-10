import pandas as pd
import numpy as np
import os.path as op
from glob import glob

main_dir = '../FEED_behav_data'
out_dir = 'data/ratings'
sub_dirs = sorted(glob(op.join(main_dir, 'raw', 'sub-*')))
to_drop_all = [
    'regular_or_catch', 'stim_path', 'trial_nr',
    'duration', 'rating_coord_deg',
]

for sub_dir in sub_dirs:
    sub = op.basename(sub_dir).split('-')[1]

    print(f"\nProcessing sub-{sub}")
    dfs = []

    ### PREPROCESSING NEUTRAL RATINGS ###
    for run in [1, 2]:
        tsv = op.join(sub_dir, f'sub-{sub}_task-neutral_run-{run}.tsv')
        if not op.isfile(tsv):
            print(f"File {tsv} does not exist!")
            continue
        df = pd.read_csv(tsv, sep='\t', index_col=0).drop(to_drop_all + ['filename', 'block', 'trial'], axis=1)
        df = df.set_index(df.trial_type).drop('rating_type', axis=1)
        df['run'] = run
        df['rep'] = run
        df['session'] = 4
        df = df.rename(columns={
            'behav_trial_nr': 'trial', 'behav_run': 'block',
            'rating_valence_norm': 'valence',
            'rating_arousal_norm': 'arousal',
            'rating_dominance': 'dominance',
            'rating_trustworthiness': 'trustworthiness',
            'rating_attractiveness': 'attractiveness'
            }
        )
        df['arousal'] = (df['arousal'] + 1) / 2  # normalize to [0, 1]
        for attr in ['dominance', 'trustworthiness', 'attractiveness']:
            df[attr] = 2 * ((df[attr] + 4) / 8) - 1 

        value_vars = ['valence', 'arousal', 'dominance', 'trustworthiness', 'attractiveness']
        id_vars = [col for col in df.columns if col not in value_vars]
        df = pd.melt(df, id_vars=id_vars, value_vars=value_vars, var_name='rating_type', value_name='rating').dropna(how='any', axis=0)
        dfs.append(df)

    df = pd.concat(dfs, axis=0)
    df = df.set_index(df.trial_type)

    df.index.name = None
    cols = ['rep', 'session', 'run', 'block', 'trial', 'trial_type', 'rating_type', 'rating', 'rating_RT']
    #cols = cols + [col for col in df.columns if col not in cols]
    df = df.loc[:, cols].sort_values(cols, axis=0)

    print("Shape neutral df: %s" % (df.shape,))
    df.to_csv(f'{out_dir}/sub-{sub}_task-neutral_ratings.tsv', sep='\t')
    ### DONE PREPROCESSING NEUTRAL RATINGS ###
    
    dfs = []
    for ses in [1, 2, 3]:
        if ses == 1:
            tsvs = sorted(glob(op.join(sub_dir, f'sub-{sub}_ses-?_task-expressive_run-?.tsv')))
            tmp = []
            for i, tsv in enumerate(tsvs):
                df = pd.read_csv(tsv, sep='\t', index_col=0)
                df['rep'] = int(op.basename(tsv).split('ses-')[1][0])
                df['run'] = int(op.basename(tsv).split('run-')[1][0])
                tmp.append(df)
            df = pd.concat(tmp, axis=0, sort=True)
        else:
            tsvs = sorted(glob(op.join(sub_dir, f'sub-{sub}_ses-{ses}_task-expressive_run-?_redo.tsv')))
            tmp = []
            for i, tsv in enumerate(tsvs):
                df = pd.read_csv(tsv, sep='\t', index_col=0)
                df['run'] = int(op.basename(tsv).split('run-')[1][0])
                df['rep'] = 1
                tmp.append(df)

            df = pd.concat(tmp, axis=0, sort=True)
        
        df = df.drop(['block', 'trial'], axis=1).rename(columns={'behav_trial_nr': 'trial', 'behav_run': 'block'})

        df_emo = df.query("rating_type == 'emotion'")
        df_emo = df_emo.drop(to_drop_all + ['rating_arousal_norm', 'rating_valence_norm'], axis=1)
        df_emo = df_emo.rename(columns={'rating_category': 'rating', 'rating_intensity_norm': 'rating_intensity'})
        nans = df_emo.loc[:, 'rating'].isna().sum()
        if nans > 0:
            print(f"WARNING: Found {nans} NaNs in the emo ratings!")
        df_emo.loc[:, 'rating'] = df_emo.loc[:, 'rating'].fillna('Geen van allen')
        
        df_circ = df.query("rating_type == 'circumplex'")
        df_circ = df_circ.drop(to_drop_all + ['rating_category', 'rating_intensity_norm'], axis=1)
        df_circ = df_circ.rename(columns={'rating_arousal_norm': 'arousal', 'rating_valence_norm': 'valence'})
        id_vars = [col for col in df_emo.columns if col not in ['rating_type', 'rating'] and col in df_circ.columns]
        df_circ = pd.melt(df_circ, id_vars=id_vars, value_vars=['valence', 'arousal'], var_name='rating_type', value_name='rating')
        df_circ = df_circ.set_index(df_circ.trial_type) 
        assert(df_circ.shape[0] == df_emo.shape[0] * 2)
        ses_df = pd.concat((df_emo, df_circ), axis=0, sort=True)
        ses_df = ses_df.set_index(ses_df.trial_type)
        ses_df.index.name = None
        dfs.append(ses_df)
        
    df = pd.concat(dfs, axis=0, sort=True)
    df = df.sort_values(['rep', 'session', 'run', 'block', 'trial'], axis=0)

    df_ses1 = df.query("session == 1")
    df_sesother = df.query("session != 1")
    print(df_sesother['data_split'].unique())
    df = pd.concat((df_ses1, df_sesother), axis=0)
    cols = ['rep', 'session', 'run', 'block', 'trial', 'data_split', 'trial_type', 'rating_type', 'rating', 'rating_intensity', 'rating_RT']
    #cols = cols + [col for col in df.columns if col not in cols]
    df = df.loc[:, cols]
    print("Shape expressive df: %s" % (df.shape,))
    df.to_csv(f'{out_dir}/sub-{sub}_task-expressive_ratings.tsv', sep='\t')
