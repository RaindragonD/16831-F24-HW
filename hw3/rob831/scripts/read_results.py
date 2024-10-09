import argparse
import glob
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

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

    def plot_results(exp):
        logdir = os.path.join('data', f"{exp}*")
        print('logdir:', logdir)
        eventfile = glob.glob(logdir)[0]

        X, Y = get_section_results(eventfile)
        for i, (x, y) in enumerate(zip(X, Y)):
            print('Iteration {:d} | Train steps: {:d} | Return: {}'.format(i, int(x), y))
            
    plot_results('q1_doubledqn')
    plot_results('q1_dqn')
    
