"""
Main script for running training session
"""
import tensorflow as tf
import json
import os

from trainer2.train import Trainer


def main(_):
    tf_config = os.environ.get('TF_CONFIG')

    # If TF_CONFIG is not available run local
    if not tf_config:
        tf.logging.info('NO TF_CONFIG!')
        return Trainer().run()

    tf_config_json = json.loads(tf_config)
    tf.logging.info(tf_config_json)

    cluster = tf_config_json.get('cluster')
    job_name = tf_config_json.get('task', {}).get('type')
    task_index = tf_config_json.get('task', {}).get('index')

    # If cluster information is empty run local
    if job_name is None or task_index is None:
        return Trainer().run()

    cluster_spec = tf.train.ClusterSpec(cluster)
    server = tf.train.Server(cluster_spec, job_name=job_name, task_index=task_index)

    if job_name == 'ps':
        server.join()
        return
    elif job_name in ['master', 'worker']:
        return Trainer().run()

    print('END MAIN')


if __name__ == '__main__':
    tf.app.run()
