# config here

import torch
from core.config import BaseAtariConfig
from core.wrap import make_atari, WarpFrame, EpisodicLifeEnv
from core.dataset import Transforms
from .EffZModel import EfficientZeroNet


class AtariAtariConfig(BaseAtariConfig):
    """
    BaseConfig gives some important functions:

      - distributional transform of values
    """
    def __init__(self):
        self.training_steps = 100000
        self.last_steps = 20000
        self.test_interval = 10000
        self.log_interval = 1000
        self.vis_interval = 1000
        self.test_episodes = 32
        self.checkpoint_interval = 100
        self.target_model_interval = 200
        self.save_ckpt_interval = 10000
        self.max_moves = 108000
        self.history_length = 400
        self.discount = 0.997
        self.dirichlet_alpha = 0.3
        self.value_delta_max = 0.01
        self.num_simulations = 50
        self.batch_size = 256
        self.td_steps = 5
        self.num_actors = 1
        self.init_zero = True
        self.clip_reward = True
        self.lr_warm_up = 0.01
        self.lr_init = 0.2
        self.lr_decay_rate = 0.1
        self.lr_decay_steps = 100000
        self.auto_td_steps_ratio = 0.3
        # lr scheduler
        self.lr_warm_up = 0.01
        self.lr_init = 0.2
        self.lr_decay_rate = 0.1
        self.lr_decay_steps = 100000
        self.auto_td_steps_ratio = 0.3
        # replay window
        self.start_transitions = 8
        self.total_transitions = 100 * 1000
        self.transition_num = 1
        # coefficient
        self.reward_loss_coeff = 1
        self.value_loss_coeff = 0.25
        self.policy_loss_coeff = 1
        self.consistency_coeff = 2
        # reward sum
        self.lstm_hidden_size = 512
        self.lstm_horizon_len = 5
        # siamese
        self.proj_hid = 1024
        self.proj_out = 1024
        self.pred_hid = 512
        self.pred_out = 1024

        super(AtariAtariConfig, self).__init__(
            test_max_moves=12000,
            # network initialization/ & normalization
            episode_life=True,
            # storage efficient
            cvt_string=True,
            image_based=True,
            # frame skip & stack observation
            frame_skip=4,
            num_stack_obs=4,)
        self.discount **= self.frame_skip  # 0.97**4 since we skip frames
        self.max_moves //= self.frame_skip
        self.test_max_moves //= self.frame_skip

        self.start_transitions = self.start_transitions * 1000 // self.frame_skip
        self.start_transitions = max(1, self.start_transitions)

        self.bn_mt = 0.1
        self.blocks = 1  # Number of blocks in the ResNet
        self.channels = 64  # Number of channels in the ResNet
        if self.gray_scale:
            self.channels = 32
        self.reduced_channels_reward = 16  # x36 Number of channels in reward head
        self.reduced_channels_value = 16  # x36 Number of channels in value head
        self.reduced_channels_policy = 16  # x36 Number of channels in policy head
        self.resnet_fc_reward_layers = [32]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [32]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [32]  # Define the hidden layers in the policy head of the prediction network
        self.downsample = True  # Downsample observations before representation network (See paper appendix Network Architecture)

    def visit_softmax_temperature_fn(self, num_moves, trained_steps):
        if self.change_temperature:
            if trained_steps < 0.5 * (self.training_steps + self.last_steps):
                return 1.0
            elif trained_steps < 0.75 * (self.training_steps + self.last_steps):
                return 0.5
            else:
                return 0.25
        else:
            return 1.0

    def set_game(self, env_name, save_video=False, save_path=None, video_callable=None):
        self.env_name = env_name
        # gray scale
        if self.gray_scale:
            self.image_channel = 1
        obs_shape = (self.image_channel, 96, 96)
        self.obs_shape = (obs_shape[0] * self.num_stack_obs, obs_shape[1], obs_shape[2])

        game = self.new_game()
        self.action_space_size = game.action_space_size

    def get_uniform_network(self):
        return EfficientZeroNet(
            self.obs_shape,
            self.action_space_size,
            self.blocks,
            self.channels,
            self.reduced_channels_reward,
            self.reduced_channels_value,
            self.reduced_channels_policy,
            self.resnet_fc_reward_layers,
            self.resnet_fc_value_layers,
            self.resnet_fc_policy_layers,
            self.reward_support.size,
            self.value_support.size,
            self.downsample,
            self.inverse_value_transform,
            self.inverse_reward_transform,
            self.lstm_hidden_size,
            bn_mt=self.bn_mt,
            proj_hid=self.proj_hid,
            proj_out=self.proj_out,
            pred_hid=self.pred_hid,
            pred_out=self.pred_out,
            init_zero=self.init_zero,
            state_norm=self.state_norm)

    def scalar_reward_loss(self, prediction, target):
        return -(torch.log_softmax(prediction, dim=1) * target).sum(1)

    def scalar_value_loss(self, prediction, target):
        return -(torch.log_softmax(prediction, dim=1) * target).sum(1)

    def set_transforms(self):
        if self.use_augmentation:
            self.transforms = Transforms(self.augmentation, image_shape=(self.obs_shape[1], self.obs_shape[2]))

    def transform(self, images):
        return self.transforms.transform(images)

    def scalar_transform(self, x):
        """ Reference from MuZero: Appendix F => Network Architecture
        & Appendix A : Proposition A.2 in https://arxiv.org/pdf/1805.11593.pdf (Page-11)
        """
        delta = self.value_support.delta
        assert delta == 1
        epsilon = 0.001
        sign = torch.ones(x.shape).float().to(x.device)
        sign[x < 0] = -1.0
        output = sign * (torch.sqrt(torch.abs(x / delta) + 1) - 1 + epsilon * x / delta)
        return output

    def inverse_reward_transform(self, reward_logits):
        return self.inverse_scalar_transform(reward_logits, self.reward_support)

    def inverse_value_transform(self, value_logits):
        return self.inverse_scalar_transform(value_logits, self.value_support)

    def inverse_scalar_transform(self, logits, scalar_support):

        """ Reference from MuZero: Appendix F => Network Architecture
        & Appendix A : Proposition A.2 in https://arxiv.org/pdf/1805.11593.pdf (Page-11)

        scalar_support is for h(reward) not for reward itself
        """
        delta = self.value_support.delta
        value_probs = torch.softmax(logits, dim=1)  # output of the model, gives distribution of h(value)
        value_support = torch.ones(value_probs.shape)
        value_support[:, :] = torch.from_numpy(np.array([x for x in scalar_support.range]))
        value_support = value_support.to(device=value_probs.device)
        value = (value_support * value_probs).sum(1, keepdim=True) / delta  # avg. keepdim to get same shape

        epsilon = 0.001
        sign = torch.ones(value.shape).float().to(value.device)
        sign[value < 0] = -1.0
        output = (((torch.sqrt(1 + 4 * epsilon * (torch.abs(value) + 1 + epsilon)) - 1) / (2 * epsilon)) ** 2 - 1)
        output = sign * output * delta

        nan_part = torch.isnan(output)
        output[nan_part] = 0.
        output[torch.abs(output) < epsilon] = 0.
        return output

    def value_phi(self, x):
        return self._phi(x, self.value_support.min, self.value_support.max, self.value_support.size)

    def reward_phi(self, x):
        return self._phi(x, self.reward_support.min, self.reward_support.max, self.reward_support.size)

    def _phi(self, x, min, max, set_size: int):
        """
        x: torch.Tensor
        it creates distribution of the input x by linear interpolation between 2 supports
        """

        delta = self.value_support.delta

        x.clamp_(min, max)
        x_low = x.floor()  # gives int (supports are ints)
        x_high = x.ceil()  # gives int
        p_high = x - x_low
        p_low = 1 - p_high

        target = torch.zeros(x.shape[0], x.shape[1], set_size).to(x.device)
        x_high_idx, x_low_idx = x_high - min / delta, x_low - min / delta
        # scatter_(dim=2, index = x_high_idx, src= p_high)
        target.scatter_(2, x_high_idx.long().unsqueeze(-1), p_high.unsqueeze(-1))
        target.scatter_(2, x_low_idx.long().unsqueeze(-1), p_low.unsqueeze(-1))
        return target

    def get_hparams(self):
        # get all the hyper-parameters
        hparams = {}
        for k, v in self.__dict__.items():
            if 'path' not in k and (v is not None):
                hparams[k] = v
        return hparams

    def set_config(self, args):
        # reset config from the args
        self.set_game(args.env)
        self.case = args.case
        self.seed = args.seed
        if not args.use_priority:
            self.priority_prob_alpha = 0
        self.amp_type = args.amp_type
        self.use_priority = args.use_priority
        self.use_max_priority = args.use_max_priority if self.use_priority else False
        self.debug = args.debug
        self.device = args.device
        self.cpu_actor = args.cpu_actor
        self.gpu_actor = args.gpu_actor
        self.p_mcts_num = args.p_mcts_num
        self.use_root_value = args.use_root_value

        if not self.do_consistency:
            self.consistency_coeff = 0
            self.augmentation = None
            self.use_augmentation = False

        if not self.use_value_prefix:
            self.lstm_horizon_len = 1

        if not self.off_correction:
            self.auto_td_steps = self.training_steps
        else:
            self.auto_td_steps = self.auto_td_steps_ratio * self.training_steps

        assert 0 <= self.lr_warm_up <= 0.1
        assert 1 <= self.lstm_horizon_len <= self.num_unroll_steps
        assert self.start_transitions >= self.batch_size

        # augmentation
        if self.consistency_coeff > 0 and args.use_augmentation:
            self.use_augmentation = True
            self.augmentation = args.augmentation
        else:
            self.use_augmentation = False

        if args.revisit_policy_search_rate is not None:
            self.revisit_policy_search_rate = args.revisit_policy_search_rate

        localtime = time.asctime(time.localtime(time.time()))
        seed_tag = 'seed={}'.format(self.seed)
        self.exp_path = os.path.join(args.result_dir, args.case, args.info, args.env, seed_tag, localtime)

        self.model_path = os.path.join(self.exp_path, 'model.p')
        self.model_dir = os.path.join(self.exp_path, 'model')
        return self.exp_path



game_config = AtariAtariConfig()
