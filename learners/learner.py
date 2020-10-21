#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 11:48:58 2017

@author: Amine Laghaout
"""

import gym
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
import tensorflow as tf
import time
import timeit

from tensorflow import keras
from . import utilities as util
from . import visualizer as vis
from . import wrangler as wra

# %% Parent classes


class Learner:

    def __init__(
            self,
            learner_dir='learner',
            default_learner_dir=['saved_learners'],
            metrics={
                x: {} for x in [
                    'explore',
                    'select',
                    'train',
                    'test',
                    'serve']},
            hyperparams=None,
            data_params=None,
            hyperparams_space=None,
            data_params_space=None,
            **kwargs):
        """
        Generic learner class.

        Parameters
        ----------
        learner_dir: str, list
            Directory where the learner is stored along with its training logs
            and performance metrics. Use ``rm -rf ./learner/`` to clean up. If
            ``learner_dir`` is a string, the directory is created under
            ``./<default_learner_dir>/``. Otherwise, if ``learner_dir`` is a
            list, that list represents the full path relative to the current
            directory (and hence ``default_learner_dir`` is ignored).
        default_learner_dir: list
            Relative path to the default parent directory.
        metrics: dict
            Default dictionary of metrics.
        hyperparams: dict
            Dictionary of hyperparameters
        data_params: dict
            Dictionary of the parameters pertaining to the data or environment.
        hyperparams_space: pd.DataFrame, None, optional
            Hyperparameter space used for model selection.
        data_params_space: pd.DataFrame, None, optional
            Data parameters space used for model selection.
        """

        # Specify the directory where the learner and its metrics are to be
        # saved.
        if isinstance(learner_dir, str):
            learner_dir = default_learner_dir + [learner_dir]
        learner_dir = os.path.join(*learner_dir)
        if not os.path.exists(learner_dir):
            os.makedirs(learner_dir)

        # Convert all the arguments to attributes.
        util.args_to_attributes(
            self, learner_dir=learner_dir, metrics=metrics,
            hyperparams=hyperparams, data_params=data_params,
            hyperparams_space=hyperparams_space,
            data_params_space=data_params_space,
            **kwargs)

        print('======================================== [start]',
              f'{self.learner_dir}\n')

        # Wrangle the data or set up the environment.
        self.wrangle()

        # Design the model and specify the metrics.
        self.design()

    def wrangle(self):
        """
        Wrangle the data or set up the environment so as to generate
        ``self.data`` or ``self.env``, respectively.
        """

        print('========== WRANGLE:')

    def design(self):
        """ Design the model. """

        print('\n========== DESIGN:')

    def explore(self):
        """
        Explore the data.
        """

        print('\n========== EXPLORE:')

    def select(self):
        """ Select the model. """

        print('\n========== SELECT:')

        if self.hyperparams_space is None:
            print('WARNING: The hyperparameter space is not specified.',
                  'Skipping the model selection.')
            return False

    def select_report(self):
        """ Report on the model selection. """

        print('\n===== Selection report:')

    def train(self):
        """ Train the model. """

        print('\n========== TRAIN:')

    def train_report(self):
        """ Report on the training. """

        print('\n===== Train report:')

    def test(self):
        """ Test the model. """

        print('\n========== TEST:')

    def test_report(self):
        """ Report on the testing. """

        print('\n===== Test report:')

    def serve(self):
        """ Serve the model. """

        print('\n========== SERVE:')

    def serve_report(self):
        """ Report on the serving. """

        print('\n===== Serve report:')

    def save(self, filename='learner.pkl'):

        # filepath = os.path.join(self.learner_dir, filename)
        # try:
        #     pickle.dump(self, open(filepath, 'wb'))
        #     print(f'Learner saved under `{filepath}`.')
        # except BaseException:
        #     print(f'WARNING: Could not save the learner under {filepath}.')

        pass

    def run(self,
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

        # Determine which stages of the pipeline to execute.
        if explore:
            self.explore()
            if pause:
                input('Press Enter to continue.')
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

        print('\n======================================== [end]',
              f' {self.learner_dir}')


class Supervised(Learner):

    pass


class Unsupervised(Learner):

    pass


class Reinforcement(Learner):

    pass

# %% Template for child classes


class LearnerChild(Learner):

    def __init__(
            self,
            learner_dir='learner_template',
            some_argument='my_default_argument',
            **kwargs):
        """
        This class is meant as a customizable template for child classes of
        ``Learner``. Replace each of the print statements with actual code so
        as to define the attributes of  ``self`` at the right places. The
        default arguments should be specified in ``self.__init__()`` and
        passed on to the parent class via ``super().__init__() below.

        This class will most often be the starting point for all other classes.

        Also note that all methods start with ``super().<the_method>()``,
        except for ``self.__init__()``, which ends with ``super().__init__()``.
        """

        print('- Perform any necessary pre-processing or validation',
              'of the arguments `kwargs`.')

        super().__init__(
            learner_dir=learner_dir,
            some_argument=some_argument,
            **kwargs)

    def wrangle(self):

        super().wrangle()

        print('- Acquire and wrangle the data -> `self.data`, or')
        # self.data = <wrangler class>()  # Acquire the raw data.
        # self.data.wrangle()  # Wrangle the data to be machine-readable.

        print('  alternatively, invoke the environment',
              '(for reinforcement learning) -> `self.env`.')
        # self.env = <wrangler class>()

        print('- Split the data set.')
        # self.data.split()

    def design(self):

        super().design()

        print('- Design the model -> `self.model`.')
        # self.model = ...

    def explore(self):

        super().explore()

        print('- Explore the data ->  `self.metrics[\'explore\']`.')
        # self.metrics['explore'] = ...

    def select(self):

        super().select()

        print('- Select a model -> `self.metrics[\'select\']`.')
        # self.model = ...  # Update the model to be the best one so far.
        # self.metrics['select'] = ...

    def select_report(self):

        super().select_report()

        print('- Report on the selection.')

    def train(self):

        super().train()

        print('- Train the model -> `self.metrics[\'train\']`.')
        # self.metrics['train'] = ...

    def train_report(self):

        super().train_report()

        print('- Report on the training.')

    def test(self):

        super().test()

        print('- Test the model -> `self.metrics[\'test\']`.')
        # self.metrics['test'] = ...

    def test_report(self):

        super().test_report()

        print('- Report on the testing`.')

    def serve(self):

        super().serve()

        print('- Serve the model -> `self.metrics[\'serve\']`.')
        # self.metrics['serve'] = ...

    def serve_report(self):

        super().serve_report()

        print('- Report on the serving.')


# %% Technology-specific parent classes.

class SupervisedKeras(Supervised):
    """ Class for SupervisedKeras learning based on Keras. """

    def explore(self, data='train'):
        """
        Show the first batch of the data (both human- and machine-readable).

        Parameters
        ----------
        data : str, optional
            Specifies what section of the data is used for exploration.
        """

        super().explore()

        # Take into account the fact that the raw data may not be available.
        try:
            print('Human-readable data:')
            self.data.view(self.data.datasets['raw'][data])
        except BaseException:
            pass

        print('Machine-readable data:')
        self.data.view(data)

    def train(self, subdir='TensorBoard'):
        """
        Parameters
        ----------
        subdir: list
            Subdirectory where the TensorBoard logs are to be stored.
        """

        super().train()

        # Tensorboard callback
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(self.learner_dir, subdir),
            histogram_freq=1, profile_batch=0, write_images=True)

        runtime = timeit.default_timer()

        self.model.fit(
            self.data.dataset['train'],
            validation_data=self.data.dataset['validate'],
            epochs=self.hyperparams['epochs'],
            callbacks=[tensorboard_callback],
        )

        self.metrics['train'] = dict(
            history=self.model.history,
            runtime=timeit.default_timer() - runtime)

    def test(self):

        super().test()

        if 'test' in self.data.dataset.keys():
            self.metrics['test'] = dict(
                zip(self.model.metrics_names + ['prediction'],
                    self.model.evaluate(self.data.dataset['test']) +
                    [self.model.predict(self.data.dataset['test'])]))
        else:
            print('WARNING: There is no \'test\' data.')

    def serve(self):

        super().serve()

        if 'serve' in self.data.dataset.keys():
            self.metrics['serve'] = self.model.predict(
                self.data.dataset['serve'])
        else:
            print('WARNING: There is no \'serve\' data.')


class ReinforcementGym(Reinforcement):

    def wrangle(self):

        super().wrangle()

        self.env = gym.make(self.env)


class QTableDiscreteGym(ReinforcementGym):

    def __init__(
            self,
            env,
            num_episodes=10_000,
            max_steps_per_episode=100,
            learning_rate=0.1,
            discount_rate=0.99,
            exploration_rate=1,
            max_exploration_rate=1,
            min_exploration_rate=0.01,
            exploration_decay_rate=0.001,
            q_table_displays=2,
            discretization=None,
            **kwargs):
        """
        Reinforcement learning using a Q-table.

        Parameters
        ----------
        env: gym environment, other
            Environment
        num_episodes: int
            Number of episodes
        max_steps_per_episode: int
            Maximum number of steps per episode
        learning_rate: float
            Learning rate in the Bellman equation
        discount_rate: float
            Discrount rate on future events
        exploration_rate: float
            Exploration (vs. exploitation rate)
        max_exploration_rate: float
            Maximum (i.e., initial) exploration rate
        min_exploration_rate: float
            Minimum (i.e., final) exploitation rate
        exploration_decay_rate: float
            Decay rate of the exploration rate
        q_table_displays: int
            Number of times that the Q-table should be displayed
        discretization: None, list
            Discretization grid of the observation space
        """

        metrics = {
            'train': {
                'reward': [None] * num_episodes,
                'exploration': [None] * num_episodes,
                'success': [None] * num_episodes,
                'steps': [max_steps_per_episode] * num_episodes},
            'test': None,
            'serve': None}

        super().__init__(
            env=env,
            learner_dir=env,
            num_episodes=num_episodes,
            max_steps_per_episode=max_steps_per_episode,
            learning_rate=learning_rate,
            discount_rate=discount_rate,
            exploration_rate=exploration_rate,
            max_exploration_rate=max_exploration_rate,
            min_exploration_rate=min_exploration_rate,
            exploration_decay_rate=exploration_decay_rate,
            q_table_displays=q_table_displays,
            discretization=discretization,
            metrics=metrics,
            **kwargs)

    def explore(self):
        """ Explore the environment. """

        print('========== EXPLORE:')

        print('Dimensionality of the action space:', self.env.action_space.n)
        print('Dimensionality of the state space:',
              self.env.observation_space.n)

    def design(self):
        """
        Design and initialize the Q-table in the case of discretized action and
        observation spaces.
        """

        # If the observation and action spaces are already discretized, simply
        # initialize the Q-table based on their dimensions.
        if self.discretization is None:
            self.q_table = np.zeros(
                (self.env.observation_space.n, self.env.action_space.n))

        # Otherwise, discretize the action and observation space.
        else:
            self.discrete_size = (self.env.observation_space.high -
                                  self.env.observation_space.low) / self.discretization
            self.q_table = np.random.uniform(
                low=self.low, high=self.high,
                size=(self.discretization + [self.env.action_space.n]))

    def train(self):

        print('========== TRAIN:')

        for episode in range(self.num_episodes):

            # Initialize the episode.
            state = self.discretize(self.env.reset())
            self.metrics['train']['reward'][episode] = []
            self.metrics['train']['exploration'][episode] = []

            # Display the Q-table at a certain frequency.
            self.display_summary(episode)

            # For each step in an episode...
            for step in range(self.max_steps_per_episode):

                # Decide whether to explore or exploit.
                action = self.explore_vs_exploit(state, episode)

                # Evaluate the next step based on the action.
                new_state, reward, done, info = self.env.step(action)
                success = self.check_success(reward, new_state)
                new_state = self.discretize(new_state)

                # Update the Q-table using the Bellman equation.
                self.update_policy(state, action, reward, new_state)

                # Render the episode at a certain frequency.
                self.render(episode)

                # Record the metrics.
                self.metrics['train']['reward'][episode] += [reward]
                self.metrics['train']['success'][episode] = success
                if done or success:
                    self.metrics['train']['steps'][episode] = step
                    break
                else:
                    state = new_state

            # Decrease the exploration rate.
            self.decay_exploration(episode)

        self.env.close()

    def train_report(self):
        """
        Parameters
        ----------
        num_batches: int
            Number of episodes in a window
        """

        print('===== Train report:')

        if not os.path.exists(os.path.join(self.learner_dir, 'train')):
            os.makedirs(os.path.join(self.learner_dir, 'train'))

        vis.plot2D(
            range(self.num_episodes),
            [sum(k) / len(k) for k in self.metrics['train']['exploration']],
            xlabel='episode', ylabel='avg. exploration rate', smooth=[51, 1],
            save_as=os.path.join(
                self.learner_dir, 'train', 'avg. exploration rate.pdf'))

        vis.plot2D(
            range(self.num_episodes),
            self.metrics['train']['success'],
            xlabel='episode', ylabel='success', smooth=[51, 1],
            save_as=os.path.join(self.learner_dir, 'train', 'success.pdf'))

        vis.plot2D(
            range(self.num_episodes),
            self.metrics['train']['steps'],
            xlabel='episode', ylabel='steps', smooth=[51, 1],
            save_as=os.path.join(self.learner_dir, 'train', 'steps.pdf'))

        vis.plot2D(
            range(self.num_episodes),
            [sum(k) / len(k) for k in self.metrics['train']['reward']],
            xlabel='episode', ylabel='avg. cumulative reward', smooth=[51, 1],
            save_as=os.path.join(
                self.learner_dir, 'train', 'avg. cumulative reward.pdf'))

        # Print the learned Q-table only if it is two-dimensional.
        if len(self.q_table.shape) == 2:
            self.display_QTable(None)

    def test(self, num_episodes=3,
             sleep_time={'episode': 1, 'step': 0.1, 'result': 3}):
        """
        Watch the agent play the best action from each state according to the
        Q-table.

        Parameters
        ----------
        num_episodes: int
            Number of episodes to serve.
        sleep_time: dict
            Dictionary of sleep times.
        """

        print('========== TEST:')

        self.metrics['test'] = {
            'steps': [self.max_steps_per_episode] * num_episodes,
            'reward': [None] * num_episodes,
            'success': [False] * num_episodes}

        for episode in range(num_episodes):

            # Initialize the episode.
            print('\n***** EPISODE ', episode + 1, '*****')
            state = self.env.reset()
            done = False
            self.render(None)
            self.metrics['test']['reward'][episode] = []
            time.sleep(sleep_time['episode'])

            for step in range(self.max_steps_per_episode):

                # Choose the action with highest Q-value for current state.
                action = self.exploit(state)

                # Take a new action.
                state, reward, done, info = self.env.step(action)
                success = self.check_success(reward, state)
                self.metrics['test']['reward'][episode] += [reward]

                # Show the current state of the environment.
                self.render(None)
                time.sleep(sleep_time['step'])

                if success or done or step == self.max_steps_per_episode - 1:
                    if success:
                        print(
                            f'**** Succeeded at step {step} with a cumulative reward of', sum(
                                self.metrics['test']['reward'][episode]), '****')
                        self.metrics['test']['success'][episode] = True
                    else:
                        print(
                            f'**** Failed at step {step} with a cumulative reward of', sum(
                                self.metrics['test']['reward'][episode]), '****')
                    time.sleep(sleep_time['result'])
                    self.metrics['test']['steps'][episode] = step
                    break

        self.env.close()

    def explore_vs_exploit(self, state, episode):

        if np.random.uniform(0, 1) > self.exploration_rate:
            action = self.exploit(state)  # Exploit.
            self.metrics['train']['exploration'][episode] += [False]
        else:
            action = self.env.action_space.sample()  # Explore.
            self.metrics['train']['exploration'][episode] += [True]

        return action

    def exploit(self, state):

        if self.discretization is None:
            return np.argmax(self.q_table[state, :])
        else:
            return np.argmax(self.q_table[state])

    def update_policy(self, state, action, reward, new_state):

        self.q_table[state, action] = \
            self.q_table[state, action] * (1 - self.learning_rate) + \
            self.learning_rate * (
                reward +
                self.discount_rate * np.max(self.q_table[new_state, :]))

    def check_success(self, reward, state):

        raise NotImplementedError

    def display_QTable(self, episode):

        plt.clf()
        plt.imshow(self.q_table.T)
        if isinstance(episode, int):
            plt.title(f'Q-table at episode {episode}')
        plt.show()

    def decay_exploration(self, episode):

        # Decay the proportion of exploration.
        self.exploration_rate = self.min_exploration_rate + \
            (self.max_exploration_rate - self.min_exploration_rate) * \
            np.exp(-self.exploration_decay_rate * episode)

    def display_summary(self, episode):

        if episode % (self.num_episodes // self.q_table_displays) == 0:
            print('\n*****EPISODE ', episode + 1, '*****')
            self.display_QTable(episode)

    def render(self, episode=None):

        if episode is None or \
                episode % (self.num_episodes // self.q_table_displays) == 0:
            self.env.render()

    def discretize(self, state):

        if self.discretization is None:
            return state
        else:
            discrete_state = (
                state - self.env.observation_space.low) / self.discrete_size
            # This tuple is used to look up the three Q-values for the
            # available actions in the q-table.
            return tuple(discrete_state.astype(np.int))

# %% Child classes for specific problem domains


class FrozenLake(QTableDiscreteGym):

    def check_success(self, reward, state):

        return reward == 1


class MountainCar(QTableDiscreteGym):

    def __init__(
            self,
            env='MountainCar-v0',
            learning_rate=0.1,
            discount_rate=0.95,
            num_episodes=7000,  # 25_000
            q_table_displays=10,
            discretization=[20, 20],
            exploration_rate=1,
            max_steps_per_episode=250,
            low=-2,
            high=0,
            decay_rate_episode=dict(start=1, end=None)):
        """
        Parameters
        ----------
        low: float
            Lowest value in the Q-table
        high: float
            Highest value in the Q-table
        """

        decay_rate_episode = decay_rate_episode
        decay_rate_episode['end'] = num_episodes // 2

        super().__init__(
            env,
            num_episodes=num_episodes,
            max_steps_per_episode=max_steps_per_episode,
            learning_rate=learning_rate,
            discount_rate=discount_rate,
            exploration_rate=exploration_rate,
            q_table_displays=q_table_displays,
            discretization=discretization,
            decay_rate_episode=decay_rate_episode,
            low=low,
            high=high)

    def explore(self):

        print("Dimensionality of the action space:", self.env.action_space.n)
        print("Dimensionality of the state space:", self.discrete_size)

    def test(self):

        pass

    def test_report(self):

        pass

    def check_success(self, reward, state):

        return state[0] >= self.env.goal_position

    def display_QTable(self, episode):

        pass

    def display_summary(self, episode):

        pass


class MountainCarOld(QTableDiscreteGym):

    def __init__(
            self,
            env='MountainCar-v0',
            learning_rate=0.1,
            discount_rate=0.95,
            num_episodes=7000,  # 25000
            q_table_displays=3000,
            discretization=[20, 20],
            exploration_rate=1,
            max_steps_per_episode=250,
            decay_rate_episode=dict(start=1, end=None)):

        self.discretization = discretization
        self.decay_rate_episode = decay_rate_episode
        self.decay_rate_episode['end'] = num_episodes // 2

        super().__init__(
            env,
            num_episodes=num_episodes,
            max_steps_per_episode=max_steps_per_episode,
            learning_rate=learning_rate,
            discount_rate=discount_rate,
            exploration_rate=exploration_rate,
            q_table_displays=q_table_displays)

    def explore(self):

        print("Dimensionality of the action space:", self.env.action_space.n)
        print("Dimensionality of the state space:", self.discrete_size)

    def design(self):

        self.discrete_size = (self.env.observation_space.high -
                              self.env.observation_space.low) / self.discretization
        self.exploration_decay_rate = self.exploration_rate / \
            (self.decay_rate_episode['end'] - self.decay_rate_episode['start'])
        self.q_table = np.random.uniform(
            low=-2, high=0, size=(self.discretization + [self.env.action_space.n]))

    def exploit(self, discrete_state):

        return np.argmax(self.q_table[discrete_state])

    def explore_vs_exploit(self, discrete_state, episode):

        if np.random.random() > self.exploration_rate:
            # Get action from Q table: Exploitation
            action = self.exploit(discrete_state)
            self.metrics['train']['exploration'][episode] += [False]
        else:
            # Get random action: Exploration
            action = np.random.randint(0, self.env.action_space.n)
            self.metrics['train']['exploration'][episode] += [True]

        return action

    def train(self):

        self.rewards_all_episodes = []

        # For each episode...
        for episode in range(self.num_episodes):
            discrete_state = self.discretize(self.env.reset())
            done = False
            rewards_current_episode = 0
            self.metrics['train']['reward'][episode] = []
            self.metrics['train']['exploration'][episode] = []

            # For each step...
            for _ in range(self.max_steps_per_episode):

                action = self.explore_vs_exploit(discrete_state, episode)

                new_state, reward, done, _ = self.env.step(action)

                new_discrete_state = self.discretize(new_state)

                # Accumulate the new reward
                rewards_current_episode += reward

                self.metrics['train']['reward'][episode] += [reward]

                if episode % self.q_table_displays == 0:
                    self.env.render()
                #new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (reward + self.discount_rate * max_future_q)

                # If simulation did not end yet after last step - update Q table
#                if not done:

                # Maximum possible Q value in next step (for new state)
                max_future_q = np.max(self.q_table[new_discrete_state])

                # Current Q value (for current state and performed action)
                current_q = self.q_table[discrete_state + (action,)]

                # And here's our equation for a new Q value for current state
                # and action
                new_q = (1 - self.learning_rate) * current_q + \
                    self.learning_rate * (reward + self.discount_rate * max_future_q)

                # Update Q table with new Q value
                self.q_table[discrete_state + (action,)] = new_q

                # Simulation ended (for any reson) - if goal position is
                # achived - update Q value with reward directly
                if new_state[0] >= self.env.goal_position:
                    #                    self.q_table[discrete_state + (action,)] = reward
                    self.q_table[discrete_state + (action,)] = 0
                    self.metrics['train']['success'][episode] = True
                    print(f'Success on episode {episode}!')
                    break

                elif done:
                    break

                discrete_state = new_discrete_state

            if episode % self.q_table_displays == 0:
                print(f'episode {episode}: {rewards_current_episode}')

            # Decaying is being done every episode if episode number is within
            # decaying range
            if self.decay_rate_episode['end'] >= episode >= self.decay_rate_episode['start']:
                self.exploration_rate -= self.exploration_decay_rate

            self.rewards_all_episodes.append(rewards_current_episode)

        self.env.close()

    def test(self, num_episodes=3,
             sleep_time={'episode': 1, 'step': 0.03, 'result': 3}):
        """
        Watch the agent play the best action from each state according to the
        Q-table.

        Parameters
        ----------
        num_episodes: int
            Number of episodes to serve.
        sleep_time: dict
            Dictionary of sleep times.
        """

        print('========== TEST:')

        self.metrics['test'] = {
            'steps': [self.max_steps_per_episode] * num_episodes,
            'reward': [None] * num_episodes,
            'success': [False] * num_episodes}

        for episode in range(num_episodes):

            # Initialize the episode.
            print('\n*****EPISODE', episode + 1, '*****')
            state = self.env.reset()
            done = False
            time.sleep(sleep_time['episode'])

            for step in range(self.max_steps_per_episode):

                # Show the current state of the environment.
                self.env.render()
                time.sleep(sleep_time['step'])

                # Choose the action with highest Q-value for current state.
                state = self.discretize(state)
                action = self.exploit(state)
                # action = np.argmax(self.q_table[state[0], state[1], :])

                # Take a new action.
                new_state, reward, done, info = self.env.step(action)

                if done or step == self.max_steps_per_episode - 1:
                    self.env.render()
                    if reward == 0 or new_state[0] >= self.env.goal_position:
                        print('**** You reached the goal! ****')
                        time.sleep(sleep_time['result'])
                    else:
                        print('**** You failed! ****')
                        time.sleep(sleep_time['result'])
                    break

                # Set the new state.
                state = new_state

        self.env.close()

    def serve(self):

        print('========== SERVE:')


class Boston(SupervisedKeras):

    def __init__(
            self, learner_dir='Boston',
            data_params=dict(
                label_name='MEDV',
                numeric_features=[
                    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
                    'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'],
                data_source=['data', 'boston.csv'],
                split_sizes={'train': 10, 'validate': 3, 'test': 2},
                batch_size=5,
                shuffle=False),
            hyperparams=dict(
                epochs=50,
                optimizer='adam',
                loss='mean_squared_error',
                metrics=['mae'],
                layers=[
                    dict(num_hidden=13, activation='relu'),
                    dict(num_hidden=1, activation=None)],
            ),
            **kwargs):
        """ This problem predicts the price of houses in Boston. """

        if isinstance(data_params['data_source'], list):
            data_params['data_source'] = os.path.join(
                *data_params['data_source'])

        super().__init__(
            learner_dir=learner_dir,
            data_params=data_params,
            hyperparams=hyperparams,
            **kwargs)

    def explore(self):

        pass

    def wrangle(self):

        # Acquire the raw data.
        self.data = wra.FromFileTF(**self.data_params)

        # Wrangle the data to be machine-readable.
        self.data.wrangle()

        # Split the data.
        self.data.split()

    def design(self):

        self.model = keras.Sequential([
            keras.layers.DenseFeatures(self.data.numeric_data)] +
            [keras.layers.Dense(layer['num_hidden'],
                                activation=layer['activation']) for layer in
             self.hyperparams['layers']])

        self.model.compile(
            loss=self.hyperparams['loss'],
            optimizer=self.hyperparams['optimizer'],
            metrics=self.hyperparams['metrics'])

    def train_report(self):

        super().train_report()

        if not os.path.exists(os.path.join(self.learner_dir, 'train')):
            os.makedirs(os.path.join(self.learner_dir, 'train'))

        # metrics = self.hyperparams['metrics']

        # vis.plot2D(
        #     self.metrics['train']['history'].epoch,
        #     (self.metrics['train']['history'].history[k] for k in
        #      metrics),
        #     xlabel='epoch', legend=metrics,
        #     show=True,
        #     save_as=os.path.join(self.learner_dir, 'train', 'metrics.pdf'))


class Heart(SupervisedKeras):

    def __init__(
            self, learner_dir='Heart',
            data_params=dict(
                label_name='target',
                data_source=['data', 'heart.csv'],
                categories=dict(thal=['fixed', 'normal', 'reversible']),
                categorical_features=['thal'],
                numeric_features=['age', 'sex', 'cp', 'trestbps', 'chol',
                                  'fbs', 'restecg', 'thalach', 'exang',
                                  'oldpeak', 'slope', 'ca'],
                split_sizes={'train': 8, 'validate': 1, 'test': 1},
                batch_size=30,
                shuffle=False),
            hyperparams=dict(
                epochs=100,
                optimizer='adam',
                loss='binary_crossentropy',
                layers=[
                    dict(num_hidden=30, activation='tanh'),
                    dict(num_hidden=20, activation='tanh'),
                    dict(num_hidden=1, activation='sigmoid')],
            ),
            **kwargs):
        """ This problem predicts the indidence of heart diseases. """

        if isinstance(data_params['data_source'], list):
            data_params['data_source'] = os.path.join(
                *data_params['data_source'])

        super().__init__(
            learner_dir=learner_dir,
            data_params=data_params,
            hyperparams=hyperparams,
            **kwargs)

    def wrangle(self):

        # Acquire the raw data.
        self.data = wra.FromFileTF(**self.data_params)

        # Wrangle the data to be machine-readable.
        self.data.wrangle()

        # Split the data.
        self.data.split()

    def design(self):

        self.model = keras.Sequential([
            keras.layers.DenseFeatures(
                self.data.numeric_data + self.data.categorical_data)] +
            [keras.layers.Dense(layer['num_hidden'],
                                activation=layer['activation']) for layer in
             self.hyperparams['layers']])

        self.model.compile(
            loss=self.hyperparams['loss'],
            optimizer=self.hyperparams['optimizer'],
            metrics=['accuracy'])

    def train_report(self):

        super().train_report()

        if not os.path.exists(os.path.join(self.learner_dir, 'train')):
            os.makedirs(os.path.join(self.learner_dir, 'train'))

        vis.plot2D(
            self.metrics['train']['history'].epoch,
            (self.metrics['train']['history'].history['accuracy'],
             self.metrics['train']['history'].history['val_accuracy']),
            xlabel='epoch', legend=['accuracy', 'val_accuracy'],
            show=True,
            save_as=os.path.join(self.learner_dir, 'train', 'metrics.pdf'))


class RotationMatrix(SupervisedKeras):

    def __init__(
            self,
            learner_dir='RotationMatrix',
            hyperparams=dict(batch_size=10, epochs=20),
            dim=2, num_examples=1000, theta=.42,
            split_sizes={'train': 50, 'validate': 20, 'test': 20, 'serve': 10},
    ):
        """
        This problem attempts to infer the matrix that rotates the feature
        vectors into the target vectors.
        """

        super().__init__(
            learner_dir=learner_dir,
            hyperparams=hyperparams,
            dim=dim, num_examples=num_examples, theta=theta,
            split_sizes=split_sizes,
        )

    def explore(self):

        pass

    def wrangle(self):

        self.data = wra.RotationMatrix(self.theta, self.num_examples)
        self.data.split(self.split_sizes)

    def design(self):

        self.layer = util.MyLayer(self.dim)

        self.model = tf.keras.Sequential([
            self.layer])

        # The compile step specifies the training configuration
        self.model.compile(
            optimizer=tf.keras.optimizers.RMSprop(0.001),
            loss=tf.keras.losses.MeanAbsolutePercentageError(),
            metrics=['mse'])

    def train_report(self):
        """ Report on the training. """

        print('===== Train report:')
        vis.plot_time_series(
            x=self.model.history.epoch,
            y_dict={x: self.model.history.history[x] for
                    x in ['mse', 'val_mse']},
            xlabel='epoch')

    def test_report(self):

        print('===== Test report:')
        print('Predicted kernel matrix:')
        print(self.layer.kernel.numpy())
        print('Actual matrix:')
        print(self.data.matrix)
