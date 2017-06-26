"""
Main script for running training session
"""
import tensorflow as tf

from trainer2 import task
from trainer2 import train
from trainer2.model import Model


def main(_):

    print('START MAIN')

    model = Model.trial()
    trainer = train.Trainer(model, task.Task())
    trainer.run()

    print('END MAIN')


if __name__ == '__main__':
    tf.app.run()
