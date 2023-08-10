from agent.modules.agents import REGISTRY as agent_REGISTRY
from agent.components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
from gym import spaces
import numpy as np


# This multi-agent controller shares parameters between agents
class BasicMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False, critic=None, target_mac=False, explore_agent_ids=None):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = agent_outputs
        # chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env,
        #                                                     test_mode=test_mode)

        chosen_actions = chosen_actions.view(ep_batch[bs].batch_size, self.n_agents, -1).detach()  # ???
        exploration_mode = "gaussian"  # TODO: might change to self.global_config
        if not test_mode:  # do exploration
            if exploration_mode == "ornstein_uhlenbeck":
                x = getattr(self, "ou_noise_state", chosen_actions.clone().zero_())
                mu = 0
                theta = getattr(self.args, "ou_theta", 0.15)
                sigma = getattr(self.args, "ou_sigma", 0.2)

                noise_scale = getattr(self.args, "ou_noise_scale", 0.3) if t_env < self.args.env_args[
                    "episode_limit"] * self.args.ou_stop_episode else 0.0
                dx = theta * (mu - x) + sigma * x.clone().normal_()
                self.ou_noise_state = x + dx
                ou_noise = self.ou_noise_state * noise_scale
                chosen_actions = chosen_actions + ou_noise
            elif exploration_mode == "gaussian":
                start_steps = getattr(self.args, "start_steps", 0)
                act_noise = getattr(self.args, "act_noise", 0.1)
                if t_env >= start_steps:
                    if explore_agent_ids is None:
                        x = chosen_actions.clone().zero_()
                        chosen_actions += act_noise * x.clone().normal_()
                    else:
                        for idx in explore_agent_ids:
                            x = chosen_actions[:, idx].clone().zero_()
                            chosen_actions[:, idx] += act_noise * x.clone().normal_()
                else:
                    if getattr(self.args.env_args, "scenario_name", None) is None or self.args.env_args[
                        "scenario_name"] in ["Humanoid-v2", "HumanoidStandup-v2"]:
                        chosen_actions = th.from_numpy(np.array(
                            [[self.args.action_spaces[0].sample() for i in range(self.n_agents)] for _ in
                             range(ep_batch[bs].batch_size)])).float().to(device=ep_batch.device)
                    else:
                        chosen_actions = th.from_numpy(np.array(
                            [[self.args.action_spaces[i].sample() for i in range(self.n_agents)] for _ in
                             range(ep_batch[bs].batch_size)])).float().to(device=ep_batch.device)

        # For continuous actions, now clamp actions to permissible action range (necessary after exploration)
        action_spaces = []
        for _ in range(self.n_agents):
            action_spaces.append(spaces.Box(low=-0.99, high=0.99, shape=(chosen_actions.shape[2],), dtype=np.float32))
        if all([isinstance(act_space, spaces.Box) for act_space in action_spaces]):
            for _aid in range(self.n_agents):
                for _actid in range(action_spaces[_aid].shape[0]):
                    chosen_actions[:, _aid, _actid].clamp_(np.asscalar(action_spaces[_aid].low[_actid]),
                                                           np.asscalar(action_spaces[_aid].high[_actid]))

        return chosen_actions

    def forward(self, ep_batch, t, hidden_states=None, select_actions=False, test_mode=False):

        # rnn based agent
        if self.args.agent in ['rnn']:
            agent_inputs = self._build_inputs(ep_batch, t)
            avail_actions = ep_batch["avail_actions"][:, t]
            agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

            # Softmax the agent outputs if they're policy logits
            if self.agent_output_type == "pi_logits":

                if getattr(self.args, "mask_before_softmax", True):
                    # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                    reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                    agent_outs[reshaped_avail_actions == 0] = -1e10

                agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
                if not test_mode:
                    # Epsilon floor
                    epsilon_action_num = agent_outs.size(-1)
                    if getattr(self.args, "mask_before_softmax", True):
                        # With probability epsilon, we will pick an available action uniformly
                        epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()

                    agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                                  + th.ones_like(agent_outs) * self.action_selector.epsilon / epsilon_action_num)

                    if getattr(self.args, "mask_before_softmax", True):
                        # Zero out the unavailable actions
                        agent_outs[reshaped_avail_actions == 0] = 0.0

        # transformer based agent
        elif self.args.agent in ['updet', 'transformer_aggregation']:
            agent_inputs = self._build_inputs_transformer(ep_batch, t)
            agent_outs, self.hidden_states = self.agent(agent_inputs,
                                                        self.hidden_states.reshape(-1, 1, self.args.emb),
                                                        self.args.enemy_num, self.args.ally_num)
        elif self.args.agent in ['trans']:
            agent_inputs = self._build_inputs_transformer(ep_batch, t)
            agent_outs, self.hidden_states = self.agent(agent_inputs,
                                                        self.hidden_states.reshape(-1, 1, self.args.emb),
                                                        self.args.enemy_num, self.args.ally_num)
            if select_actions:
                return agent_outs

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        if self.args.agent in ['rnn']:
            self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav
        elif self.args.agent in ['updet', 'transformer_aggregation']:
            self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, 1, 1, -1)
        elif self.args.agent in ['trans']:
            self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, 1, 1, -1)

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions"][:, t]))
            else:
                inputs.append(batch["actions"][:, t - 1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _build_inputs_transformer(self, batch, t):
        # currently we only support battles with marines (e.g. 3m 8m 5m_vs_6m)
        # you can implement your own with any other agent type.
        inputs = []
        raw_obs = batch["obs"][:, t]
        arranged_obs = raw_obs
        obs_size = arranged_obs.shape[2]
        agent_num = arranged_obs.shape[1]
        # reshaped_obs = arranged_obs.view(-1, 1 + (self.args.enemy_num - 1) + self.args.ally_num, self.args.token_dim)
        reshaped_obs = arranged_obs.contiguous()  # .view(-1, 1, obs_size)
        inputs.append(reshaped_obs)
        inputs = th.cat(inputs, dim=1).cuda()
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
