import argparse
import glob
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt

def get_section_results(file):
    """
        requires tensorflow==1.12.0
    """
    X = []
    Y = []
    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Train_EnvstepsSoFar':
                X.append(v.simple_value)
            elif v.tag == 'Train_AverageReturn':
                Y.append(v.simple_value)
        if len(X) > 120:
            break
    return X, Y

if __name__ == '__main__':

    def plot_results(exp, la):
        logdir = os.path.join('data', f"{exp}*", "events*")
        eventfiles = glob.glob(logdir)
        
        Ys = []
        for eventfile in eventfiles:
            X, Y = get_section_results(eventfile)
            Ys.append(Y)
        mean = np.mean(np.array(Ys), axis=0)
        std = np.std(np.array(Ys), axis=0)
        

        plt.errorbar(X, mean, yerr=std, label=la, capsize=5)
        plt.xlabel('Train Steps')
        plt.ylabel('Average Return')
        plt.title(f'Training Results for {exp}')

    plt.legend()
    plt.show()

    plot_results('q1_doubledqn', 'Double DQN')
    plot_results('q1_dqn', 'DQN')
    
