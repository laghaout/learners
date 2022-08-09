#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 11:48:58 2017

@author: Amine Laghaout
"""

import os
import json
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers


def version_table(print2screen=True):
    """
    This function returns the version numbers of the various pieces of software
    with which this module was tested.
    Notes
    -----
    In order for Hyperopt 0.1 to work, ``networkx`` had to be downgraded by
    running ``pip install networkx==1.11``. This is due to a bug that arises
    with Hyperopt when version 2.0 of ``networkx`` is installed.
    Also include:
        - conda install plotly
    Parameters
    ----------
    print2screen : bool
        Print the version table to screen (``True``) or return it as a
        dictionary (``False``)?
    Returns
    -------
    version_table : dict
        Dictionary containing the version table
    """

    import cpuinfo  # python -m pip install -U py-cpuinfo
    import platform

    from dcor import __version__ as dco_version
    from matplotlib import __version__ as plt_version
    from numpy import __version__ as np_version
    from pandas import __version__ as pd_version
    from sklearn import __version__ as sk_version
    from sys import version_info
    from tensorflow import __version__ as tf_version

    version_table = {
        'Python': ('3.9.10', '.'.join(str(v) for v in version_info[0:3])),
        'TensorFlow.': ('2.9.1', tf_version),
        'NumPy': ('1.22.4', np_version),
        'matplotlib': ('3.5.2', plt_version),
        'sklearn': ('1.1.1', sk_version),
        'PyQt5': ('5.6.2', None),
        'pandas': ('1.4.3', pd_version),
        'dcor': ('0.5.6', dco_version),
        'OS': ('Linux-5.10.0-14-amd64-x86_64-with-glibc2.31',
               platform.platform()),
        'CPU': ('Intel(R) Core(TM) i7-7500U CPU @ 2.70GHz',
                cpuinfo.get_cpu_info()['brand_raw']),
        'CUDA': ('8.0.44', None),
        'GPU': ('NVIDIA GeForce GTX', None)}

    if print2screen:

        # Maximum length of the software names
        pad = max(map(lambda x: len(x), version_table))

        # Print the table.
        print('software'.rjust(pad), ': baseline', sep='')
        print(''.rjust(pad), '  current', sep='')
        for k in sorted(version_table.keys()):
            print(k.rjust(pad), ': ', version_table[k][0], sep='')
            print(''.rjust(pad), '  ', version_table[k][1], sep='')

    return version_table


def set_argv(defaults, argv):
    """
    This utility function is used to overwrite the default list of arguments
    with those that are passed via ``sys.argv``.

    Parameters
    ----------
    defaults: list
        List of default arguments
    argv: list
        List of arguments to replace the defaults

    Return
    ------
    defaults: list
        Updated list of arguments
    """

    # Note that the first argument is skipped since it is merely the name of
    # the calling function.
    defaults[:len(argv) - 1] = argv[1:]

    return defaults


def args_to_attributes(obj, **kwargs):
    """
    Assign the items of the dictionaries ``default_args`` and ``kwargs`` as
    attributes to the object ``obj``.
    Parameters
    ----------
    obj : object
        Object to which attributes are to be assigned
    kwargs : dict
        Dictionary of attributes to overwrite the defaults.
    """

    [obj.__setattr__(k, kwargs[k]) for k in kwargs.keys()]

    return obj


def view_with_pandas(df, index=None, encoding='utf-8'):
    """
    View a ``tf.data.Dataset`` batch as a ``pandas.DataFrame`` by converting
    all the byte strings into regular Python strings
    """

    def decode(x):
        try:
            x = x.decode(encoding)
        except Exception:
            pass
        return x

    df = df.applymap(decode)

    df[index] = df[index].astype('str')
    if index is not None:
        df.set_index(index, inplace=True)

    return df


def select_rows(df, specs):
    """
    Select the rows of the data frame ``df`` as per the columns specified by
    the key-value pairs in ``specs``.

    Parameters
    ----------
    df: pandas.DataFrame
        Data frame to to be filtered by column values.
    specs: dict
        Dictionary of column values.

    Return
    ------
    df: pandas.DataFrame
        The input data frame where the rows match the column specifications in
        ``specs``.
    """

    for k in specs.keys():
        if isinstance(specs[k], tuple) or isinstance(specs[k], list) or \
                isinstance(specs[k], set):
            df = df.loc[df[k] in specs[k]].copy()
        else:
            df = df.loc[df[k] == specs[k]].copy()

    return df


def rw_data(path, data=None, params=None):

    extension = path.split('.')[-1].lower()

    # Read
    if data is None:

        print(f'Reading `{path}`.')

        path = open(path, 'rb')

        if extension in ('yaml', 'yml'):
            import yaml
            if params is None:
                params = dict(Loader=yaml.FullLoader)
            data = yaml.load(path, **params)
        elif extension in ('pickle', 'pkl'):
            import pickle
            data = pickle.load(path)
        elif extension in ('json'):
            import json
            data = json.load(path)
        elif extension in ('hdf5', 'h5', 'hdf'):
            pass
        elif extension in ('csv'):
            pass
        else:
            print('WARNING: No file format specified.')

        path.close()

        return data

    # Write
    else:

        print(f'Writing to `{path}`.')

        path = open(path, 'wb')

        if extension in ('yaml', 'yml'):
            import yaml
            yaml.dump(data, path, default_flow_style=False)
        elif extension in ('pickle', 'pkl'):
            import pickle
            pickle.dump(data, path)
        elif extension in ('json'):
            import json
            json.dump(data, fp=path)
        elif extension in ('hdf5', 'h5', 'hdf'):
            pass
        elif extension in ('csv'):
            pass
        else:
            print('WARNING: No file format specified.')
            return False

        path.close()

        return True


def score_to_label(score, threshold, smaller_label, larger_label):
    """
    Convert a score to either one of two labels based on a threshold.

    Parameters
    ----------
    score : float
        Score
    threshold : float
        Threshold
    smaller_label : str
        Label to return if the score is strictly smaller than the threshold
    larger_label : TYPE
        Label to return if the score is equal or greater than the threshold

    Returns
    -------
    str
        Label
    """
    return smaller_label if score < threshold else larger_label


def dict_json(x, y=None):

    # Save the dictionary ``x`` to a JSON ``y``.
    if isinstance(x, dict):
        if isinstance(y, str):
            y = [y]
        json.dump(x, fp=open(*y, 'w'))

    # Extract the dictionary ``y`` from the JSON ``x``.
    else:
        if isinstance(x, str):
            x = [x]

        return json.load(open(x, 'rb'))


def fetch_object(path_from_ref_dir, attributes=None, ref_dir=['/', 'home']):
    """
    Fetch an object from its context.

    Parameters
    ----------
    path_from_ref_dir : list
        Path to the object from the reference directory.
    attribute: str, list of str, None
        If a string, return only the attribute of the object which is labeled
        with that string. If it is a list, return all the attributes in that
        list as a dictrionary. The default is None, in which case the whole
        object is returned.
    ref_dir : TYPE, optional
        Path to the reference directory. The default is ['/', 'home'].

    Returns
    -------
    saved_object : object
        Objecty to be returned
    """

    curr_dir = os.getcwd()

    # Temporarily move to home to view the pickled object in its context.
    os.chdir(os.path.join(*ref_dir))

    # Load the pickled object.
    saved_object = rw_data(os.path.join(*path_from_ref_dir))

    # Move back to the current map you're implementing
    os.chdir(curr_dir)

    if attributes is None:
        return saved_object
    elif isinstance(attributes, str):
        return getattr(saved_object, attributes)
    elif isinstance(attributes, list):
        return {attribute: getattr(saved_object, attribute)
                for attribute in attributes}


def assemble_dataframe(batch, label, label_name='label'):
    """
    Parameters
    ----------
    batch: tf.Tensor
        Features tensor
    label: tf.Tensor
        Labels tensor
    label_name: str
        Name of the label (i.e., target)

    Return
    ------
    batch: pandas.DataFrame
        Data frame which assembles the batch with its labels into one

    TODO: The try~except block is supposed to accommodate the fact that the
    elements_spec is different depending on how the data was generated. Find a
    less hacky way to do this.
    """

    # Use this for tf.Tensors
    try:

        batch = pd.DataFrame(batch.numpy())
        label = pd.DataFrame(label.numpy())
        batch = pd.concat(
            [batch, label], axis=1, sort=False)

    # Use this for make_csv_dataset()
    except Exception:

        batch = pd.DataFrame(batch)
        label = pd.DataFrame(
            {label_name: label}, index=range(len(label)))
        batch = pd.concat(
            [batch, label], axis=1, sort=False)

    batch = pd.DataFrame(batch)

    return batch


class PackNumericFeatures(object):
    """
    https://www.tensorflow.org/tutorials/load_data/csv#data_preprocessing
    """

    def __init__(self, names):
        self.names = names

    def __call__(self, features, labels):
        numeric_features = [features.pop(name) for name in self.names]
        numeric_features = [tf.cast(feat, tf.float32) for
                            feat in numeric_features]
        numeric_features = tf.stack(numeric_features, axis=-1)
        features['numeric'] = numeric_features

        return features, labels


class MyLayer(layers.Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_shape[1], self.output_dim),
            initializer='uniform',
            trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.kernel)

    def get_config(self):
        base_config = super(MyLayer, self).get_config()
        base_config['output_dim'] = self.output_dim
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
