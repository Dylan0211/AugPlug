"""
@file: controller.py
@date: 2023/12/20 17:46
@desc: 
"""
import collections
import os
import random
import pandas as pd
import torch
import torch.nn.functional as F
import AugPlug.utils
from AugPlug.TCN import TemporalConvNet


class PolicyNet(torch.nn.Module):
    def __init__(self, num_input_type, output_size_dict, device, args):
        super(PolicyNet, self).__init__()
        self.args = args
        self.enc_dim = args.controller_enc_dim
        self.hidden_dim = args.controller_hidden_dim
        self.output_size_dict = output_size_dict
        self.num_op_per_policy = args.num_op_per_policy
        self.device = device

        self.tcn_embedding = TemporalConvNet(input_channels=[1, 16, 32], output_channels=[16, 32, 64], num_blocks=3, kernel_sizes=[24, 7, 4], dilations=[1, 24, 24 * 7])
        self.input_embedding = torch.nn.Embedding(num_input_type, 16)
        self.lstm = torch.nn.LSTMCell(64 + 16, 64 + 16)

        self.decoder_dict = dict()
        for mode, output_size in output_size_dict.items():
            decoder = torch.nn.Linear(64 + 16, output_size).to(device)
            self.decoder_dict.update({mode: decoder})
        self.num_size = len(output_size_dict)
        self.num_param_list = list(self.output_size_dict.values())

    def forward(self, state, list_fixed_actions=None):
        """
        :param state: information of X_1, X_2, concept drift type, params of child network
        :return: prob_list: softmax probabilities for sequence of parameters [type_1, prob_1, mag_1, type_2, ...]
        """
        # note: process state with TCN embedding layers
        state = torch.tensor(state, dtype=torch.float32).reshape(1, 1, -1).to(self.device)
        state = self.tcn_embedding(state)[:, :, -1].squeeze().reshape(1, -1)

        list_selected_log_probs = []
        list_actions = []
        for u in range(self.args.U[self.args.when_to_update[0]]):
            selected_log_probs = []
            actions = []
            input = torch.zeros((1, 16), dtype=torch.float32).to(self.device)
            hidden = (torch.zeros((1, 64 + 16), dtype=torch.float32).to(self.device),
                      torch.zeros((1, 64 + 16), dtype=torch.float32).to(self.device))
            for i in range(3 * self.num_op_per_policy):
                if i != 0:
                    input = self.input_embedding(input)

                lstm_input = torch.concat((input, state), dim=-1)
                hx, cx = self.lstm(lstm_input, hidden)

                # get selected action, prob and log_prob
                out = self.decoder_dict[i % self.num_size](hx)
                prob = F.softmax(out, dim=-1)
                log_prob = F.log_softmax(out, dim=-1)

                if list_fixed_actions is None:
                    action = prob.multinomial(num_samples=1).data
                else:
                    action = list_fixed_actions[u][i]
                selected_log_prob = log_prob.gather(1, action)
                selected_log_probs.append(selected_log_prob)
                actions.append(action)

                # update input, hidden and ptr
                mode = i % self.num_size
                input = torch.tensor([action.item() + sum(self.num_param_list[:mode])]).to(self.device)
                hidden = (hx, cx)
            selected_log_probs = torch.cat(selected_log_probs).sum().view(-1, 1)
            list_selected_log_probs.extend(selected_log_probs)
            list_actions.append(actions)
        return list_selected_log_probs, list_actions


# controller based on PPO
class Controller:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.output_size_dict = {
            0: len(self.args.augment_types),
            1: len(self.args.magnitude_types),
            2: len(self.args.probability_types)
        }
        self.output_size_list = [
            len(self.args.augment_types),
            len(self.args.magnitude_types),
            len(self.args.probability_types),
        ] * self.args.num_op_per_policy
        self.num_input_type = sum(self.output_size_list)

        # actor
        self.actor = PolicyNet(self.num_input_type, self.output_size_dict, device, args).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), self.args.controller_actor_lr)

        # buffer (collect stream data)
        self.buffer = pd.DataFrame()

    def take_action(self, state):
        list_selected_log_probs, list_actions = self.actor(state)
        list_policy_dict = []
        for u in range(self.args.U[self.args.when_to_update[0]]):
            policy_dict = {}
            actions = list_actions[u]
            for j in range(self.args.num_op_per_policy):
                operation = {
                    'type': self.args.augment_types[actions[3 * j].item()],
                    'mag': self.args.magnitude_types[actions[3 * j + 1].item()],
                    'prob': self.args.probability_types[actions[3 * j + 2].item()]
                }
                policy_dict[j] = operation
            list_policy_dict.append(policy_dict)
        return list_policy_dict, list_actions, list_selected_log_probs

    def update(self, state, list_actions, reward, list_old_log_probs):
        reward = torch.tensor(reward, dtype=torch.float32).view(-1, 1).to(self.device)
        old_log_prob = torch.cat(list_old_log_probs).sum().view(-1, 1)

        for k in range(self.args.controller_update_epochs):
            list_new_log_probs, _ = self.actor(state, list_fixed_actions=list_actions)
            new_log_prob = torch.cat(list_new_log_probs).sum().view(-1, 1)

            ratio = torch.exp(new_log_prob - old_log_prob)
            surr1 = ratio * reward
            surr2 = torch.clamp(ratio, 1 - self.args.controller_eps, 1 + self.args.controller_eps) * reward

            actor_loss = torch.mean(-torch.min(surr1, surr2))
            # print(f'        $ actor loss: {actor_loss.cpu().detach().item()}')

            self.actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor_optimizer.step()

    # def reset_buffer(self):
    #     self.buffer = pd.DataFrame()
    #
    # def append_data_to_buffer(self, df_stream):
    #     self.buffer = pd.concat([self.buffer, df_stream])
    #     self.buffer.reset_index(drop=True, inplace=True)
    #
    # def output_source_data_for_transformation(self):
    #     if self.buffer.shape[0] < self.args.min_length_of_training_data:
    #         n = 1
    #         while n * self.buffer.shape[0] < self.args.min_length_of_training_data:
    #             df_src = pd.concat((self.buffer, self.buffer))
    #             n += 1
    #     else:
    #         df_src = self.buffer.loc[self.buffer.shape[0] - self.args.min_length_of_training_data:, :].copy()
    #     df_src.reset_index(drop=True, inplace=True)
    #     return df_src
