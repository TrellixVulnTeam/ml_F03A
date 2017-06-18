import tensorflow as tf


class Trainer(object):
    """Class for model training."""

    def __init__(self, model, task):
        self.model = model
        self.task = task

    def run(self):
        """
        Run trainer.
        """
