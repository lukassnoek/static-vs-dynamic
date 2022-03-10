import sys
import logging
import os.path as op
import pandas as pd
import numpy as np
from tqdm import tqdm 
from copy import copy
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from noiseceiling import reduce_repeats


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(funcName)-8.8s] [%(levelname)-7.7s]  %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler()
    ]
)


DU2EN = dict(
    Verrassing='surprise',
    Verdrietig='sadness',
    Bang='fear',
    Walging='disgust',
    Blij='happiness',
    Boos='anger'
)


class DataLoader:
    
    def __init__(self, sub='01', data_dir=None, rnd_seed=42, log_level=20):
        """ Initializes a DataLoader object.

        Parameters
        ----------
        sub : str
            Subject ID (zero-padded)
        data_dir : str
            Path to data directory
        rnd_seed : int
            Random seed for train-test set split
        """
        self.sub = sub
        self.y = None
        self.X = None
        self.target_name = None 

        if data_dir is None:
            data_dir = op.abspath('data')
            if not op.isdir(data_dir):
                raise ValueError(f"Directory {data_dir} does not exist.")

        self.data_dir = data_dir
        self.rnd_seed = rnd_seed
        # To go from strings to integers and back
        self.le = LabelEncoder().fit(['happiness', 'surprise', 'fear', 'sadness', 'disgust', 'anger'])
        self.log = logging.getLogger(__name__)
        self.log.setLevel(log_level)

    def load_y(self, target='emotion', data_split='train', filter_gva=True):
        """ Loads the target variable (y). 
        
        Parameters
        ----------
        target : str
            Name of target variable ("emotion", "valence", or "arousal")
        data_split : str
            Either "train" or "test"
        filter_gva : bool
            Whether to remove the "geen van allen" (none of all) ratings
        """
        # e.g., "emotion" or "arousal"
        self.target_name = target

        f = op.join(self.data_dir, 'ratings', f'sub-{self.sub}_task-expressive_ratings.tsv')
        df = pd.read_csv(f, sep='\t', index_col=0)
        df = df.query("rating_type == @target")  # filter rating type
        n_orig = df.shape[0]
        if data_split == 'train':
            df = df.query("data_split == 'train'")
        elif data_split == 'test':
            df = df.query("data_split == 'test' or data_split == 'test_extra'")
        
        self.log.info(f"Removed {n_orig - df.shape[0]} test trials (N = {df.shape[0]}).")
        
        if filter_gva and target == 'emotion':  # remove "geen van allen" (none of all)
            n_orig = df.shape[0]
            df = df.query("rating != 'Geen van allen'")
            n_remov = n_orig - df.shape[0]
            self.log.info(f"Removed {n_remov} 'Geen van allen' trials (N = {df.shape[0]}).")

        if target == 'emotion':
            with pd.option_context('mode.chained_assignment', None):  # suppress stupid warning
                df.loc[:, 'rating'] = df.loc[:, 'rating'].replace(DU2EN)  # translate
                df.loc[:, 'rating'] = self.le.transform(df['rating'])  # string to integer

        self.y = df.copy()['rating']
        self.rating_df = df

    def load_X(self, feature_set, n_comp=50, standardize=False, reduce_repeats=True):
        """ Loads in predictors/independent variables (X).

        Parameters
        ----------
        feature_set : str/list/tuple
            Name of one or more feature-sets to use as predictors
        n_comp : int
            Number of components 
        standardize : bool
            Whether to standardize (0 mean, 1 std) the data.
        """

        if not isinstance(feature_set, (list, tuple)):
            feature_set = (feature_set,)

        if self.y is None:
            raise ValueError("Call load_y before load_X!")

        X = []
        for fs in feature_set:  # load in 1 or more feature-sets
            if fs in ['+', '#']:
                continue

            self.log.info(f"Loading feature-set {fs}.")
            path = op.join(self.data_dir, 'features', f'{fs}.tsv')
            df = pd.read_csv(path, sep='\t', index_col=0)
            
            if 'sub' in df.columns:
                df = df.query(f"sub == 'sub-{self.sub}'").drop('sub', axis=1)

            # Make sure X (df) and y align
            df = df.loc[self.y.index, :]
            
            if 'data_split' in df.columns:
                df = df.drop('data_split', axis=1)

            if 'PCA' in fs:
                df = df.iloc[:, :n_comp]

            X.append(df)

        if '#' in feature_set:
            if not np.all(X[1].sum(axis=1) == 1):
                X[1]['icept'] = 1
            b = np.linalg.lstsq(X[1].to_numpy(), X[0].to_numpy(), rcond=None)[0]
            X[0].loc[:, :] = X[0].to_numpy() - X[1].to_numpy() @ b 
            X = X[0]
        else:
            X = pd.concat(X, axis=1)

        categorical = True if self.target_name == 'emotion' else False

        if reduce_repeats:
            from noiseceiling.utils import reduce_repeats as red_rep
            try:
                X, self.y = red_rep(X, self.y, use_index=True, categorical=categorical)
            except ValueError:
                self.log.warning(f"No repeats for {self.sub}!")

        if not categorical:
            self.y = self.y.astype(float)
            self.y[self.y.isna()] = 0

        if standardize:
            X = (X - X.mean(axis=0)) / X.std(axis=0)
                    
        self.X = X
        self.log.info(f"Shape X: {self.X.shape}.")

    def return_Xy(self):
        """ Returns the labels (y) and predictors (X). """
        assert(self.X.index.equals(self.y.index))
        return self.X, self.y
