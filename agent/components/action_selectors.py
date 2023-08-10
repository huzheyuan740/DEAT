import torch as th
from torch.distributions import Categorical
from .epsilon_schedules import DecayThenFlatSchedule
import numpy as np

REGISTRY = {}


class MultinomialActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")  # "linear"
        self.epsilon = self.schedule.eval(0)
        self.test_greedy = getattr(args, "test_greedy", True)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        masked_policies = agent_inputs.clone()
        masked_policies[avail_actions == 0.0] = 0.0

        self.epsilon = self.schedule.eval(t_env)

        if test_mode and self.test_greedy:
            picked_actions = masked_policies.max(dim=2)[1]
        else:
            picked_actions = Categorical(masked_policies).sample().long()

        return picked_actions


REGISTRY["multinomial"] = MultinomialActionSelector


class EpsilonGreedyActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        # Assuming agent_inputs is a batch of Q-Values for each agent bav

        if test_mode:
            # Greedy action selection only
            self.epsilon = 0.0
        else:
            self.epsilon = self.schedule.eval(t_env)

        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0.0] = -float("inf")

        # print("masked_q_values:", masked_q_values)
        random_numbers = th.rand_like(agent_inputs[:, :, 0])
        # print("self.epsilon:", self.epsilon)
        pick_random = (random_numbers < self.epsilon).long()
        # random_actions = th.randint(0, 2, size=(avail_actions.shape[0], avail_actions.shape[1])).long().cuda()
        # print("pick_random:", pick_random)
        random_actions = Categorical(avail_actions.float()).sample().long()
        # print("max_result_index:", masked_q_values.max(dim=2)[1])
        # print("(1 - pick_random):", (1 - pick_random))

        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        # print("picked_actions:", picked_actions)
        return picked_actions


REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector
