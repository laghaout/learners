#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 11:48:58 2017

@author: Amine Laghaout
"""

import os
import time

try:
    USER = ''
    from . import wrangler as wra
    from . import utilities as util
except BaseException:
    import getpass
    USER = getpass.getuser()
    import wrangler as wra
    import utilities as util

# %% Learner class


class Learner:

    def __init__(
            self,
            lesson_dir=['lessons'],
            data_params=dict(),
            hyperparams=dict(),
            hyperparams_space=None,
            verbose=1,
            report={
                x: dict() for x in
                ['wrangle', 'explore', 'select', 'train', 'test', 'serve']},
            **kwargs):
        """
        Generic learner class.

        Parameters
        ----------
        lesson_dir: list, tuple
            Relative path of the directory where the lesson is stored.
        default_lesson_dir: list
            Relative path to the default parent directory.
        report: dict
            Default dictionary of reports from the various stages.
        hyperparams: dict
            Dictionary of hyperparameters
        data_params: dict
            Dictionary of the parameters pertaining to the data or environment.
        hyperparams_space: pd.DataFrame, None, optional
            Hyperparameter space used for model selection.
        """

        # Specify the directory where the learner and its metrics are to be
        # saved.
        if isinstance(lesson_dir, list) or isinstance(lesson_dir, tuple):
            lesson_dir = os.path.join(*lesson_dir)
        if not os.path.exists(lesson_dir):
            os.makedirs(lesson_dir)

        # Convert all the arguments to attributes.
        util.args_to_attributes(
            self, lesson_dir=lesson_dir, report=report,
            hyperparams=hyperparams, data_params=data_params,
            hyperparams_space=hyperparams_space, verbose=verbose, **kwargs)

    def wrangle(self):
        """ Prepare the data. """

        if self.verbose is not False:
            print('\n========== WRANGLE:')

        delta_tau = time.time()

        # Acquire, validate, and shuffle the raw data.
        self.data = wra.Wrangler(**self.data_params)

        # Wrangle (i.e., engineer features), split, and normalize.
        self.report['wrangle'] = self.data()

        self.data.split()

        self.data.normalize()

        self.report['wrangle']['delta_tau'] = time.time() - delta_tau

    def design(self):
        """ Design the model. """

        if self.verbose is not False:
            print('\n========== DESIGN:')

        self.model = None

    def explore(self):
        """ Explore the data. """

        if self.verbose is not False:
            print('\n========== EXPLORE:')

        delta_tau = time.time()

        self.report['explore'] = self.data.explore()

        self.report['explore']['delta_tau'] = time.time() - delta_tau

    def select(self):
        """ Select the model. """

        if self.verbose is not False:
            print('\n========== SELECT:')

        if self.hyperparams_space is None:
            print('WARNING: The hyperparameter space is not specified.',
                  'Skipping the model selection.')
            self.report['select'] = None
        else:

            self.report['select']['delta_tau'] = time.time()

    def select_report(self):
        """ Report on the model selection. """

        if self.verbose is not False:
            print('===== Selection report:')

    def train(self):
        """ Train the model. """

        if self.verbose is not False:
            print('\n========== TRAIN:')

        self.report['train']['delta_tau'] = time.time()

    def train_report(self):
        """ Report on the training. """

        if self.verbose is not False:
            print('===== Train report:')

    def test(self):
        """ Test the model. """

        if self.verbose is not False:
            print('\n========== TEST:')

        self.report['test']['metrics'] = None

        self.report['test']['delta_tau'] = time.time()

    def test_report(self):
        """ Report on the testing. """

        if self.verbose is not False:
            print('===== Test report:')

    def serve(self):
        """ Serve the model. """

        if self.verbose is not False:
            print('\n========== SERVE:')

        self.report['serve']['delta_tau'] = time.time()

    def serve_report(self):
        """ Report on the serving. """

        if self.verbose is not False:
            print('===== Serve report:')

    def save(self, timestamp=True):

        if self.verbose is not False:
            print('\n========== SAVE:')

        if timestamp is True:
            timestamp = round(time.time())

        # TODO: This causes the error "Can't pickle local object". Find a way
        #       to save the learner object and not just the report.
        # try:
        #     util.rw_data(
        #         os.path.join(self.lesson_dir, f'learner{timestamp}.pkl'),
        #         self)
        #     if self.verbose is not False:
        #         print('✓ Saved the learner.')
        # except BaseException:
        #     util.rw_data(
        #         os.path.join(self.lesson_dir, f'report{timestamp}.pkl'),
        #         self.report)
        #     try:  # Keras model
        #         self.model.save(
        #             os.path.join(
        #                 self.lesson_dir,
        #                 f'model{timestamp}'))
        #     except BaseException:
        #         util.rw_data(
        #             os.path.join(self.lesson_dir, f'model{timestamp}.pkl'),
        #             self)
        #     if self.verbose is not False:
        #         print('✓ Saved the report and the model.')

    def __call__(self,
                 explore=True, select=True, train=True, test=True, serve=True,
                 pause=False):
        """
        Run the various stages of the learning.

        Parameters
        ----------
        explore: bool
            Explore the data?
        select: bool
            Select the model?
        train: bool
            Train the model?
        test: bool
            Test the model?
        serve: bool
            Serve the model?
        pause: bool
            Pause in between runs?
        """

        if self.verbose is not False:
            print('======================================== [start]',
                  f'{self.lesson_dir}')

        self.wrangle()

        if explore:
            self.explore()
            self.data.consolidate()
            self.data.save()
            if pause:
                input('Press Enter to continue.')

        self.design()

        if select:
            self.select()
            self.select_report()
            if pause:
                input('Press Enter to continue.')
        if train:
            self.train()
            self.train_report()
            if pause:
                input('Press Enter to continue.')
        if test:
            self.test()
            self.test_report()
            if pause:
                input('Press Enter to continue.')
        if serve:
            self.serve()
            self.serve_report()
        self.save()

        if self.verbose is not False:
            print('======================================== [end]',
                  f' {self.lesson_dir}')

# %% Run locally.

# if len(USER) > 0:
#     learner = Learner()
#     learner()
#     report = learner.report
