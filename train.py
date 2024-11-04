"""
@file: train___expr.py
@date: 2023/12/11 11:17
@desc: 
"""
import copy
from AugPlug.utils import *
from config import *
from model.LSTM import LSTM
from AugPlug.controller import Controller
from AugPlug.auto_augment import augment_with_policy
from data.dataloader import *
import os
import logging
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import warnings
warnings.filterwarnings("ignore")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def train_controller(args, controller, train_data_dict, device):
    # logs
    logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.addHandler(console_handler)

    # train
    print('start controller training')
    for epoch in range(args.num_episodes):
        print('-' * 50)
        print('{} th episode'.format(epoch + 1))
        print('-' * 50)

        cumulative_reward = 0.
        for building_name, source_data_dict in train_data_dict.items():


            print(f'* Building Name: {building_name}')

            # retrieve update request index and corresponding version of deployed model
            training_list_for_each_update = source_data_dict[1]

            for idx in range(len(training_list_for_each_update)):
                # record start time
                start_time = time.time()

                # get list of training components
                df_train = training_list_for_each_update[idx][0].copy()
                df_val = training_list_for_each_update[idx][1].copy()
                model_before_update = copy.deepcopy(training_list_for_each_update[idx][2])
                model_after_update = copy.deepcopy(training_list_for_each_update[idx][3])

                train_x, train_y = get_train_xy(df_train, args.model_seq_length)
                val_x, val_y = get_val_xy(df_val, args.model_seq_length)

                # get state
                state = np.array(df_train['nor_load'])

                # sample policy
                list_policy_dict, list_actions, list_selected_log_probs = controller.take_action(state)

                # perform augmentation
                aug_train_x, aug_train_y = augment_with_policy(train_x, train_y, list_policy_dict, args)

                # prepare dataloader
                train_loader = get_train_loader(aug_train_x, aug_train_y, args)
                val_loader = get_val_loader(val_x, val_y, args)

                # update and eval child, get reward
                model_before_update.update(train_loader, device, args)
                val_loss_with_augment = model_before_update.validate(val_loader, device)
                val_loss_without_augment = model_after_update.validate(val_loader, device)

                reward = val_loss_without_augment - val_loss_with_augment
                cumulative_reward += reward
                print('\n    - Reward: {:.2f}'.format(reward))

                # record end time
                end_time = time.time()
                print('time: ', end_time - start_time)

                # update controller
                controller.update(state, list_actions, reward, list_selected_log_probs)

                # save controller
                with open(f'model/saved_models/controller_LSTM_{args.when_to_update[0]}_{str(args.when_to_update[1])}'
                          f'_{args.how_to_update}_{args.min_length_of_chunk_in_data_stream}.pkl', 'wb') as w:
                    pickle.dump(controller, w)

        print(f'- cumulative reward: {cumulative_reward}')
        logging.info(f'cumulative reward: {cumulative_reward}')


if __name__ == '__main__':
    # settings
    args = get_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)

    # model and training data setup
    BLF_model = LSTM(args).to(device)
    BLF_model.load_state_dict(torch.load('model/saved_models/lstm_OIE_24h.pt'))
    train_data_dict, inference_data_dict = get_data_from_genome(BLF_model, device, args)

    # start training
    controller = Controller(args, device)
    # with open(f'model/saved_models/controller_LSTM_{args.when_to_update[0]}_{str(args.when_to_update[1])}'
    #           f'_{args.how_to_update}_{args.min_length_of_chunk_in_data_stream}.pkl', 'rb') as r:
    #     controller = pickle.load(r)
    train_controller(args, controller, train_data_dict, device)

