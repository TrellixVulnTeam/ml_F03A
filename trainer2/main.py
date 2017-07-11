"""
Main script for running training session
"""
import tensorflow as tf

from trainer2.config import config_factory
from trainer2.train import Trainer


def main(_):
    config = config_factory()

    if config.job_name == 'ps':
        config.server.join()
        return

    Trainer(runtime_config=config).run()


if __name__ == '__main__':
    tf.app.run()
