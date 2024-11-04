import numpy as np
import random
import pandas as pd


def augment_with_policy(train_x, train_y, list_policy_dict, args):
    # define ranges of type, magnitude, and probability
    prob_ranges = np.linspace(0, 1, len(args.probability_types))
    mag_ranges = {
        'jitter': np.linspace(0.1, 0.3, len(args.magnitude_types)),
        'scale': np.concatenate((np.linspace(-1.0, -0.5, len(args.magnitude_types) // 2),
                                 np.linspace(0.5, 1.0, len(args.magnitude_types) // 2))),
        'shift': np.concatenate((np.linspace(-0.5, -0.1, len(args.magnitude_types) // 2),
                                 np.linspace(0.1, 0.5, len(args.magnitude_types) // 2))),
        'smooth': np.linspace(0.1, 0.3, len(args.magnitude_types)),
    }
    func = {
        'jitter': jitter,
        'scale': scale,
        'shift': shift,
        'smooth': smooth,
    }

    # perform data transformation
    list_of_augmented_train_x, list_of_augmented_train_y = [], []
    for i, policy_dict in enumerate(list_policy_dict):
        # print policy information
        print(f'    - Sampled policy {i + 1}:', end='')
        for p in policy_dict.items():
            print(p, end='')
        print()

        # perform composition of transformations
        for v in range(args.V[args.how_to_update]):
            tmp_train_x, tmp_train_y = train_x.copy(), train_y.copy()
            for k in range(tmp_train_x.shape[0]):
                for j in range(args.num_op_per_policy):
                    if random.random() < prob_ranges[policy_dict[j]['prob']]:
                        type = policy_dict[j]['type']
                        mag = policy_dict[j]['mag']

                        # for load, augment trace in x and y
                        tmp_train_x[k, :, 0] = np.clip(func[type](tmp_train_x[k, :, 0], mag_ranges[type][mag]), 0, 1)
                        tmp_train_y[k, :] = np.clip(func[type](tmp_train_y[k, :], mag_ranges[type][mag]), 0, 1)
            list_of_augmented_train_x.append(tmp_train_x)
            list_of_augmented_train_y.append(tmp_train_y)
    aug_train_x = np.concatenate(list_of_augmented_train_x, axis=0)
    aug_train_y = np.concatenate(list_of_augmented_train_y, axis=0)

    return aug_train_x, aug_train_y


def shift(x, mag):
    output = x + mag
    return np.array(output)

def jitter(x, mag):
    scale = np.random.normal(loc=0., scale=mag, size=x.shape)
    output = [x[i] * (1 + scale[i]) for i in range(x.shape[0])]
    return np.array(output)

def scale(x, mag):
    # this_max, this_min = np.percentile(x, 0.9), np.percentile(x, 0.3)
    # x = np.array([(this_x - this_min)/(this_max - this_min) for this_x in x])
    output = x * (1 + mag)
    # output = [out * (this_max - this_min) + this_min for out in output]
    return np.array(output)

def smooth(x, mag):
    # moving average (magnitude is the size of window)
    def exponential_moving_average(data, alpha):
        ema = [data[0]]
        for i in range(1, len(data)):
            ema.append(alpha * data[i] + (1 - alpha) * ema[i - 1])
        return np.array(ema)
    output = exponential_moving_average(x, mag)
    return np.array(output)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from config import get_args
    args = get_args()

    x = [random.uniform(0., 1.) for _ in range(24)]
    output = scale(x, mag=10)
    a = 0
