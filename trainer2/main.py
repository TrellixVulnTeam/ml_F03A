"""
Main script for running training session
"""
import tensorflow as tf

try:
    from trainer2.train import Trainer
    from trainer2.config import config_factory
except ImportError:
    from train import Trainer
    from config import config_factory


def main(_):
    config = config_factory()

    if config.job_name == 'ps':
        config.server.join()
        return

    Trainer(config=config).run()


if __name__ == '__main__':
    tf.app.run()
