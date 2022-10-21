#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 11:48:58 2017

@author: Amine Laghaout

This module contains the most common high-level data wranglers.
"""

import tensorflow as tf

try:
    from . import utilities as util
except BaseException:
    import utilities as util


class Wrangler:

    def __init__(self, data_source=None, verbose=True, **kwargs):
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
        util.args_to_attributes(
            self, data_source=data_source, verbose=verbose, **kwargs)

        # Load (or generate) the raw data. Note that the wrangling per se,
        # i.e., the conversion into the machine-readable data, is not done in
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

        self.shuffle()

    def acquire(self):
        """
        Data acquisition. This is where the raw, human-readable data is
        assembled.
        """

        if self.verbose is not False:
            print('===== Acquiring the data…')

    def validate(self):
        """ Validate the data. """

        if self.verbose is not False:
            print('===== Validating the data…')

        return True

    def explore(self):
        """ Explore the data, either visually or statistically. """

        if self.verbose is not False:
            print('===== Exploring the data…')

        return dict(stats=None)

    def __call__(self):
        """
        Data wrangling. This transforms the raw, human-readable data to the
        (typically numerical) machine-readable data that can be ingested by
        machine learning algorithms. This can be thought of as the most basic
        (and often only) layer of feature engineering.
        """

        if self.verbose is not False:
            print('===== Wrangling the data…')

        # ... wrangling logic goes here...

        return dict(stats=None)

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

        if self.verbose is not False:
            print('===== Splitting the data…')

    def shuffle(self):
        """ Shuffle the datasets. """

        if self.verbose is not False:
            print('===== Shuffling the data…')

    def normalize(self):
        """ Normalize the datasets. """

        if self.verbose is not False:
            print('===== Normalizing the data…')

    def consolidate(self):

        if self.verbose is not False:
            print('===== Consolidating the data…')

    def save(self):
        """ Save the data object. """

        if self.verbose is not False:
            print('===== Saving the wrangler object…')
