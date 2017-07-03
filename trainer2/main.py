"""
Main script for running training session
"""
import tensorflow as tf

from trainer2.train import Trainer
from trainer2.util import get_config


def main(_):
    config = get_config()

    if config.job_name == 'ps':
        config.server.join()
    elif config.job_name in ['', 'worker', 'master']:
        Trainer(config=config).run()
    else:
        raise ValueError('Invalid task_type: {}'.format(config.job_name))

if __name__ == '__main__':
    tf.app.run()
