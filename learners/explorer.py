#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 20:10:52 2021

@author: ala
"""

import collections
import matplotlib.pyplot as plt
import os
import pandas as pd
from pandas_profiling import ProfileReport
import seaborn as sns


sns.set(font_scale=1.2)


class Explorer:

    def __init__(
            self,
            source,
            usecols=None,
            delimiter=None,
            index_col=None,
            **kwargs):

        self.source_dir = os.path.split(source)[0]

        self.source = source
        self.usecols = usecols
        self.delimiter = delimiter
        self.index_col = index_col

        # Set the kwargs as attributes.
        [self.__setattr__(k, v) for k, v in kwargs.items()]

        if (isinstance(self.source, list) or
                isinstance(self.source, tuple)):
            self.source = os.path.join(*self.source)

        print(f'Loading `{self.source}`...')
        self.data = pd.read_csv(
            self.source, delimiter=self.delimiter, nrows=self.nrows,
            usecols=self.usecols)

        # Shuffle the rows?
        if hasattr(self, 'shuffle'):
            if self.shuffle:
                self.data = self.data.sample(frac=1).reset_index(drop=True)

        # If ``usecols`` was passed, ensure the columns are in the same order.
        if self.usecols is not None:
            self.data = self.data[self.usecols]

        print(self.data.head())
        print('Shape:', self.data.shape)

    def __call__(self, profile_report='report'):
        """
        Extract various statistics and generic descriptions of the data.

        Parameters
        ----------
        profile_report_path : str
            Pathname for saving the profile report.
        """

        self.report = pd.concat([
            pd.DataFrame({'filled': self.data.notna().mean()}),
            pd.DataFrame({'dtypes': self.data.dtypes}),
            self.data.describe(
                include='all', datetime_is_numeric=True).T], axis=1)

        if isinstance(profile_report, str):
            self.profile_report = ProfileReport(self.data)
            self.profile_report.to_file(os.path.join(
                *[self.source_dir, f'{profile_report}.html']))
            # Note that JSONs are up to 10 times heavier than the HTML reports.
            # self.profile_report.to_file(os.path.join(
            #     *[self.source_dir, f'{profile_report}.json']))

    def select_category(self, category, value):

        return self.data[self.data[category] == value]

    @staticmethod
    def bar_plot(analysis, pdfpath, bbox_to_anchor=(1, 1.17)):

        # Prepare the analysis in a format that can be read by `sns.barplot()`.
        analyses = []
        for col in analysis.columns:
            analyses += [analysis[col].reset_index(level=0).rename(
                columns={col: 'coverage'})]
            analyses[-1]['category'] = col
        analysis = pd.concat(analyses)

        ax = sns.barplot(
            data=analysis,
            x='coverage',
            y='index',
            hue='category',
            palette='Blues_d')
        ax.set_xlabel('Coverage')
        ax.set_ylabel('Columns')
        ax.legend(
            loc='upper right', bbox_to_anchor=bbox_to_anchor, ncol=2)

        plt.savefig(
            os.path.join(*pdfpath),
            bbox_inches='tight')
        plt.show()
        plt.close()

    @staticmethod
    def find_duplicates(iterable):

        return [item for item, count in collections.Counter(
            iterable).items() if count > 1]

    def engineer(self):

        pass
