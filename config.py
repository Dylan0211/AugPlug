import numpy as np
import argparse


def get_args():
    parser = argparse.ArgumentParser('Time-series AutoAugment')
    # forecasting models
    parser.add_argument('--model_structure_name', type=str, default='lstm')
    parser.add_argument('--model_seq_length', type=int, default=24)
    parser.add_argument('--model_input_dim', type=int, default=33)
    parser.add_argument('--model_sparse_output_dim', type=int, default=64)
    parser.add_argument('--model_hidden_dim', type=int, default=128)
    parser.add_argument('--model_output_dim', type=int, default=1)
    parser.add_argument('--model_num_layers', type=int, default=1)
    parser.add_argument('--model_enc_hid_dim', type=int, default=128)
    parser.add_argument('--model_dec_hid_dim', type=int, default=128)
    parser.add_argument('--model_sparse_lstm_lambda', type=float, default=1e-3)
    parser.add_argument('--model_sparse_ed_lambda', type=float, default=1e-5)

    parser.add_argument('--when_to_update', type=list, default=['periodically', 24 * 7 * 4])
    # parser.add_argument('--when_to_update', type=list, default=['triggered', 0.3])
    # parser.add_argument('--how_to_update', type=str, default='retrain')
    parser.add_argument('--how_to_update', type=str, default='finetune')

    # dataset
    parser.add_argument('--load_data_dir', type=str, default='D:\\PythonProjects\\raw_data\\Genome\\meters\\cleaned\\electricity_cleaned.csv')
    parser.add_argument('--temp_data_dir', type=str, default='D:\\PythonProjects\\raw_data\\Genome\\weather\\weather.csv')
    parser.add_argument('--tmp_data_dir', type=str, default='./data/tmp_pkl_data')
    parser.add_argument('--selected_site_list', type=list, default=['Panther', 'Fox', 'Rat', 'Bear', 'Bull', 'Peacock', 'Cockatoo'])
    parser.add_argument('--selected_category_list', type=list, default=['office', 'education', 'public', 'assembly', 'lodging'])
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--min_length_of_chunk_in_data_stream', type=int, default=24 * 7 * 4)

    # search space
    parser.add_argument('--augment_types', type=list, default=['jitter', 'scale', 'shift', 'smooth'], help='all searched policies')
    parser.add_argument('--magnitude_types', type=list, default=range(10))
    parser.add_argument('--probability_types', type=list, default=range(10))
    parser.add_argument('--num_op_per_policy', type=int, default=3)

    # controller
    parser.add_argument('--U', type=dict, default={'periodically': 3, 'triggered': 1})  # transformation parameter U
    parser.add_argument('--V', type=dict, default={'retrain': 3, 'finetune': 1})  # transformation parameter V
    parser.add_argument('--controller_input_dim', type=int, default=1)
    parser.add_argument('--controller_state_dim', type=int, default=1)
    parser.add_argument('--controller_enc_dim', type=int, default=16)
    parser.add_argument('--controller_hidden_dim', type=int, default=64)
    parser.add_argument('--controller_actor_lr', type=float, default=1e-4)
    parser.add_argument('--controller_critic_lr', type=float, default=1e-3)
    parser.add_argument('--controller_gamma', type=float, default=0.98)
    parser.add_argument('--controller_lmbda', type=float, default=0.95)
    parser.add_argument('--controller_eps', type=float, default=0.2)
    parser.add_argument('--controller_update_epochs', type=int, default=80)
    parser.add_argument('--num_cd_type', type=int, default=3)

    # training
    parser.add_argument('--model_training_epochs', type=int, default=5)
    parser.add_argument('--model_learning_rate', type=float, default=1e-3)
    parser.add_argument('--num_episodes', type=int, default=200)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--mode', type=str, default='train')

    arguments = parser.parse_args()
    print(arguments)
    return arguments
