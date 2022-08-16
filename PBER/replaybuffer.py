import ray
import time

import numpy as np
from core.history import GameHistory
from pber.config import BaseAtariConfig


@ray.remote  # remote -> distributed to other cpus to complete then retrieve
class ReplayBuffer(object):
    """Reference : DISTRIBUTED PRIORITIZED EXPERIENCE REPLAY
    Algo. 1 and Algo. 2 in Page-3 of (https://arxiv.org/pdf/1803.00933.pdf
    """

    def __init__(self, config: BaseAtariConfig):
        self.back_factor = config.back_factor
        self.back_step = config.back_step
        self.config = config
        # self.batch_queue = batch_queue
        self.batch_size = config.batch_size
        self.keep_ratio = 1

        self.model_index = 0
        self.model_update_interval = 10

        self.buffer = []
        self.priorities = np.array([])
        self.betas = np.array([])
        self.raw_betas = []  # not modifiable
        self.game_look_up = []

        self._eps_collected = 0  # episodes collected
        self._trans_collected = 0
        self.base_idx = 0  # recorder of how many games deleted
        self._alpha = self.config.priority_prob_alpha
        self.clear_time = 0
        self.transition_top = config.rb_transition_size

    def save_pools(self, pools):
        # save a list of game histories
        for (game, value_prefix, priorities) in pools:
            # Only append end game
            # if end_tag:
            self.save_game(game, value_prefix, priorities)

    def save_game(self, game: GameHistory, value_prefix, priorities=None):  # TODO
        """
        Save a game history block
        Parameters
        ----------
        game: Any
            a game history block, a sequence !
        priorities: possible-list
            the priorities corresponding to the transitions in the game history
        """
        # if self.get_total_len() >= self.config.total_transitions:
        #     return

        valid_len = len(game)
        self._eps_collected += 1
        self._trans_collected += valid_len

        if priorities is None:
            max_prio = max(self.priorities) if self.buffer else 1  # initialize priorities to be maximum
            self.priorities = np.concatenate((self.priorities, [max_prio for _ in range(valid_len)]))
        else:
            assert len(game) == len(priorities), " priorities should be of same length as the game steps"
            self.priorities = np.concatenate((self.priorities, priorities.reshape(-1)))

        self.betas = np.concatenate((self.betas, [self.back_factor ** (valid_len - 1 - l_) for l_ in range(valid_len)]))
        self.raw_betas += [self.back_factor ** (valid_len - 1 - l_) for l_ in range(valid_len)]

        # directly add game object to the buffer. Later, it's convenient to get n-step return
        self.buffer.append((game, value_prefix))  # [s0,s1,s2] [S0,S1,S2,S3] game s != S; look_up is counting

        # game's (start_idx, length) in the self.buffer len(game_look_up) = len(priority) = len(transitions)
        self.game_look_up += [(self.base_idx + len(self.buffer) - 1, step_pos) for step_pos in range(valid_len)]

    def get_game(self, idx):
        # return a game
        # game_id: the history, game_pos: the step in that history
        game_id, game_pos = self.game_look_up[idx]
        game_id -= self.base_idx
        game = self.buffer[game_id]
        return game  # a sequence of game transitions

    def prepare_batch_context(self, batch_size, beta):
        # ray.get(replay_buffer.prepare_batch_context.remote(10,1))
        """Prepare a batch context that contains:
        game_lst:               a list of game histories
        game_pos_lst:           transition index in game (relative index)
        indices_lst:            transition index in replay buffer
        weights_lst:            the weight concerning the priority
        make_time:              the time the batch is made (for correctly updating replay buffer when data is deleted)
        Parameters
        ----------
        batch_size: int
            batch size
        beta: float
            the parameter in PER for calculating the priority
        """
        assert beta > 0

        total = self.get_total_len()

        probs = (self.priorities * self.betas) ** self._alpha

        probs /= probs.sum()
        # sample data, by indices
        # for specific transition ?
        indices_lst = np.random.choice(total, batch_size, p=probs, replace=False)
        # importance sampling weights
        weights_lst = (total * probs[indices_lst]) ** (-beta)

        weights_lst /= weights_lst.max()

        game_lst = []
        game_pos_lst = []

        for idx in indices_lst:
            game_id, game_pos = self.game_look_up[idx]
            game_id -= self.base_idx
            game = self.buffer[game_id]  # (game_history, prefix)

            game_lst.append(game)
            game_pos_lst.append(game_pos)

        make_time = [time.time() for _ in range(len(indices_lst))]

        # indices_lst is the list of transitions idx inside the game_look_up list
        # game_lst gives the list of where the game is
        # game_pos_lst gives where the transition in the game
        # weights_lst gives the weight for importance sampling
        context = (game_lst, game_pos_lst, indices_lst, weights_lst, make_time)
        return context

    def update_priorities(self, indices_lst, new_priorities, make_time):
        """

        :param indices_lst: global index of the transition
        :param new_priorities: new values
        :param make_time:
        :return:
        """
        # update the priorities for data still in replay buffer
        for i, (idx, prio) in enumerate(zip(indices_lst, new_priorities)):
            if make_time[i] > self.clear_time:
                self.priorities[idx] = prio

                # self.betas[idx] = 1            # TODO update betas
                self.betas[idx] = self.raw_betas[idx]
                _, pos = self.game_look_up[idx]
                # back-propagate beta
                for back_ in range(pos):
                    if back_ > self.back_step:
                        break
                    self.betas[idx-1-back_] = max(self.back_factor ** back_, self.betas[idx - 1 - back_])
                # reset the current beta

    def remove_to_fit(self):
        # remove some old data if the replay buffer is full.
        current_size = self.size()
        total_transition = self.get_total_len()
        # if total_transition > self.transition_top:
        if total_transition > self.transition_top:
            index = 0
            for i in range(current_size):
                total_transition -= len(self.buffer[i][0])  # length of game
                if total_transition <= self.transition_top * self.keep_ratio:
                    index = i
                    break

            if total_transition >= self.batch_size:
                self._remove(index + 1)
                # print(f'prune buffer..trans={c_num}->{self.get_total_len()}/{self.transition_top}')

    def _remove(self, num_excess_games):
        # delete game histories
        excess_games_steps = sum([len(game) for game, _ in self.buffer[:num_excess_games]])
        del self.buffer[:num_excess_games]
        self.priorities = self.priorities[excess_games_steps:]
        self.betas = self.betas[excess_games_steps:]
        self.raw_betas = self.raw_betas[excess_games_steps:]

        del self.game_look_up[:excess_games_steps]
        self.base_idx += num_excess_games

        self.clear_time = time.time()

    def clear_buffer(self):
        del self.buffer[:]
        self.priorities = np.array([])
        self.betas = np.array([])
        self.raw_betas = []
        self.game_look_up = []

        self._eps_collected = 0  # episodes collected
        self._trans_collected = 0
        self.base_idx = 0  # recorder of how many games deleted
        self.clear_time = 0

        self.clear_time = time.time()

    def size(self):
        # number of games
        return len(self.buffer)

    def episodes_collected(self):
        # number of collected histories
        return self._eps_collected

    def transitions_collected(self):
        return self._trans_collected

    def get_batch_size(self):
        return self.batch_size

    def get_priorities(self):
        return self.priorities

    def get_total_len(self):
        # number of transitions
        return len(self.priorities)

    def is_full(self):
        return len(self.priorities) > self.transition_top

