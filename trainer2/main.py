"""
Main script for running training session
"""
import tensorflow as tf

from trainer2.train import Trainer
from trainer2.util import config_factory


def main(_):
    config = config_factory()
    Trainer(config=config).run()

    # if config.job_name == 'ps':
    #     config.server.join()
    # elif config.job_name in ['master', 'worker']:
    #
    # else:
    #     raise ValueError('Invalid task_type: {}'.format(config.job_name))


if __name__ == '__main__':
    tf.app.run()
