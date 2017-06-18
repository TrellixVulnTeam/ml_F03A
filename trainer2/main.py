"""
Main script for running training session
"""
import tensorflow as tf

from trainer2 import task
from trainer2 import train
from trainer2 import model

def main(_):
    print('Init training...')

    trainer = train.Trainer(model.Model.trial(), task.Task())
    trainer.run()


if __name__ == '__main__':
    tf.app.run()
