#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 11:48:58 2017

@author: Amine Laghaout

This module contains the most common high-level data wranglers.
"""

import tensorflow as tf

from . import utilities as util


class Wrangler:

    def __init__(self, data_source=None, **kwargs):
        """
        Generic data/environment wrangler class.

        Parameters
        ----------
        data_source: str, None
            Specification of the data source (e.g., pathname to a CSV file,
            log-in credentials to a database, etc).
        """

        # This lists serves to store the dataset at the successive stages of
        # the wrangling. The first stage is typically the raw data straight
        # from the source, whereas the very last is the machine-readable
        # dataset, i.e., after operations such as one-hot encoding,
        # normalization, etc.
        self.datasets = dict()

        # Convert all the arguments to attributes.
        util.args_to_attributes(self, data_source=data_source, **kwargs)

        # Load (or generate) the raw data. Note that the wrangling per se,
        # i.e., the conversion into the machine-readable data, is not run in
        # ``__init__()``. It should be run separately after the creation of the
        # wrangler object. The reason is that the user may want to first
        # inspect the raw data before performing the transformation into
        # machine-readable data. The option of storing both the raw, human-
        # readable data and the machine-readable data (as well as any inter-
        # mediate transformation) may only be feasible if the data is evaluated
        # lazily. In such cases, the sequence of data transformations may be
        # stored in a list ``self.datasets`` where the first element is the raw
        # data and the last---equivalent to ``self.dataset``---is typically
        # the most refined, machine-readable data.
        self.acquire()

        # Validate the raw data.
        assert self.validate()

    def acquire(self):
        """
        Data acquisition. This is where the raw, human-readable data is
        assembled.
        """

        pass

    def validate(self):
        """ Validate the data. """

        return True

    def explore(self):
        """ Explore the data, either visually or statistically. """

        pass

    def wrangle(self):
        """
        Data wrangling. This transforms the raw, human-readable data to the
        (typically numerical) machine-readable data that can be ingested by
        machine learning algorithms. This can be thought of as the most basic
        (and often only) layer of feature engineering.
        """

        pass

    def view(self):
        """ View one or several batches of data. """

        pass

    def split(self, split_sizes=None):
        """
        Split the data sets into the sections specified by ``split_sizes``. As
        a result ``self.dataset`` as well as each element in ``self.datasets``
        is split accordingly.

        Parameter
        ---------
        split_sizes: dict, None
            Dictionary which specifies the sizes of the various sections to
            split thhe data into. The values in the dictionary can refer to
            numbers of examples, numbers of batches, or percentages thereof.
            (The exact choice is implementation-dependent.)
        """

        pass

    def shuffle(self):
        """ Shuffle the datasets. """

        pass

    def stratify(self):
        """ Stratify the datasets. """

        pass

    def normalize(self):
        """ Normalize the datasets. """

        pass


class WranglerPD(Wrangler):

    def view(self, dataset=None):

        if dataset is None:
            print(self.dataset.head())
        elif isinstance(dataset, str):
            print(self.dataset[dataset].head())


class FromFilePD(WranglerPD):

    pass


class WranglerTF(Wrangler):

    def view(self, dataset=None, batch_num=0, num_batches=5,
             return_list=False, print2screen=True):
        """
        View specific batches from the dataset. This assumes to use of
        ``tf.data.Dataset``.

        Parameters
        ----------
        dataset: None, str, tf.data.Dataset
            Dataset to view. If ``None`` use the default dataset
            ``self.dataset``, if a string, use it to specify the key to the
            dictionary ``self.dataset[dataset]``. Otherwise, use the dataset
            explicitely passed as ``dataset``.
        num_batches: int
            Number of batches to view
        batch_num: int, None
            Index of the batch to display (starting from zero). If ``None``,
            return all batches.
        return_list: bool
            Return the list of ``num_batch`` batches?
        print2screen: bool
            Print the successive batches to the screen?

        Return
        ------
        batches: list
            A list of of ``pandas.DataFrame`` corresponding to the
            ``num_batches`` retrieved.
        """

        # If no list of batches is to be returned, don't bother loading more
        # batches than is necessary to return the ``batch_num``th batch.
        if return_list is False and batch_num is not None:
            num_batches = batch_num + 1

        batches = []

        # View the default dataset. This assumed that the dataset has not been
        # split.
        if dataset is None:
            dataset = self.dataset

        # View one of the four splits of the dataset.
        elif isinstance(dataset, str):
            dataset = self.dataset[dataset]

        # If the dataset is a tuple...
        if isinstance(dataset.element_spec, tuple):

            # ... of two elements, then assume that we are dealing with
            # supervised learning.
            if len(dataset.element_spec) == 2:

                # For each batch,
                for batch, label in dataset.take(num_batches):

                    # store the current batch as a pandas data frame.
                    batches += [util.assemble_dataframe(
                        batch, label, self.label_name)]

        if print2screen:

            # Once ``num_batches`` are retrieved, either print every one
            if batch_num is None:
                for batch_num, batch in enumerate(batches):
                    print(f'Batch {batch_num}:\n', batches[batch_num])

            # or only print the one that is specified at position
            # ``batch_num``.
            else:
                print(f'Batch {batch_num}:\n', batches[batch_num])

        if return_list:
            return batches

    def split_helper(self, data, split_sizes):

        dataset = dict()

        for key in split_sizes.keys():
            dataset[key] = data.take(split_sizes[key])
            data = data.skip(split_sizes[key])

        return dataset

    def split(self, split_sizes=None):
        """
        Split the dataset.

        Parameter
        ---------
        split_sizes: dict of int
            Dictionary that specifies the number of batches to be allocated to
            training, valdiation, test, etc (or watever splitting is
            specified).
        """

        if isinstance(split_sizes, dict):
            self.split_sizes = split_sizes

        # Split the main dataset.
        self.dataset = self.split_helper(self.dataset, self.split_sizes)

        # Split all other datasets.
        if hasattr(self, 'datasets'):
            for k in self.datasets.keys():
                self.datasets[k] = self.split_helper(
                    self.datasets[k], self.split_sizes)


class FromFileTF(WranglerTF):

    def __init__(
            self,
            data_source=None,
            label_name=None,
            batch_size=5,
            num_epochs=1,
            na_value='?',
            ignore_errors=True,
            shuffle=False,
            field_delim=',',
            use_quote_delim=True,
            shuffle_buffer_size=10000,
            **kwargs):
        """ Generic data wrangler based on CSV files """

        super().__init__(
            batch_size=batch_size,
            num_epochs=num_epochs,
            data_source=data_source,
            label_name=label_name,
            na_value=na_value,
            ignore_errors=ignore_errors,
            shuffle=shuffle,
            use_quote_delim=use_quote_delim,
            field_delim=field_delim,
            shuffle_buffer_size=shuffle_buffer_size, **kwargs)

    def acquire(self, **kwargs):

        self.dataset = tf.data.experimental.make_csv_dataset(
            self.data_source,
            batch_size=self.batch_size,
            label_name=self.label_name,
            na_value=self.na_value,
            num_epochs=self.num_epochs,
            ignore_errors=self.ignore_errors,
            shuffle=self.shuffle,
            use_quote_delim=self.use_quote_delim,
            field_delim=self.field_delim,
            shuffle_buffer_size=self.shuffle_buffer_size,
            **kwargs)

    def wrangle(self):

        self.datasets['raw'] = self.dataset

        # Numeric data
        if hasattr(self, 'features_numeric'):
            self.dataset = self.dataset.map(
                util.PackNumericFeatures(self.numeric_features))
            self.data_numeric = tf.feature_column.numeric_column(
                'numeric', shape=[len(self.numeric_features)])
            self.data_numeric = [self.data_numeric]

        # Categorical data
        if hasattr(self, 'features_categorical'):
            self.data_categorical = []
            for feature, vocab in self.categories.items():
                cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
                    key=feature, vocabulary_list=vocab)
                self.data_categorical.append(
                    tf.feature_column.indicator_column(cat_col))

        self.datasets['wrangled'] = self.dataset
