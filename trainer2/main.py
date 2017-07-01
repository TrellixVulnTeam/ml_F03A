"""
Main script for running training session
"""
import tensorflow as tf

from trainer2.train import Trainer
from trainer2.util import get_config


def main(_):
    config = get_config()
    trainer = Trainer(config=config)
    trainer.run()

if __name__ == '__main__':
    tf.app.run()
