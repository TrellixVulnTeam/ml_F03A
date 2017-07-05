"""
Main script for running training session
"""
import tensorflow as tf

from trainer2.train import Trainer
from trainer2.util import config_factory


def main(_):
    config = config_factory()

    if config.job_name == 'ps':
        config.server.join()
        return

    Trainer(config=config).run()


if __name__ == '__main__':
    tf.app.run()
