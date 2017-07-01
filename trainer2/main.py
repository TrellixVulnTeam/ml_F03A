"""
Main script for running training session
"""
import tensorflow as tf
import json
import os

from trainer2 import task
from trainer2 import train
from trainer2.model import Model


def main(_):
    tf_config = os.environ.get('TF_CONFIG')

    # If TF_CONFIG is not available run local
    if not tf_config:
        tf.logging.info('NO TF_CONFIG!')

    tf_config_json = json.loads(tf_config)
    tf.logging.info(tf_config_json)

    cluster = tf_config_json.get('cluster')
    job_name = tf_config_json.get('task', {}).get('type')
    task_index = tf_config_json.get('task', {}).get('index')

    model = Model.trial()
    trainer = train.Trainer(model, task.Task())
    trainer.run()

    print('END MAIN')


if __name__ == '__main__':
    tf.app.run()
