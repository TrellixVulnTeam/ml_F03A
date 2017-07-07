
import threading
import time

from trainer2.util import log_fn


class GlobalStepWatcher(threading.Thread):
    """A helper class for globe_step.

    Polls for changes in the global_step of the model, and finishes when the
    number of steps for the global run are done.
    """

    def __init__(self, sess, global_step_op,
                 start_at_global_step, end_at_global_step):
        threading.Thread.__init__(self)
        self.sess = sess
        self.global_step_op = global_step_op
        self.start_at_global_step = start_at_global_step
        self.end_at_global_step = end_at_global_step

        self.start_time = 0
        self.start_step = 0
        self.finish_time = 0
        self.finish_step = 0
        self._value = 0

    def run(self):
        while self.finish_time == 0:
            time.sleep(.25)
            global_step_val, = self.sess.run([self.global_step_op])
            self._value = global_step_val
            if self.start_time == 0 and global_step_val >= self.start_at_global_step:
                log_fn('Starting real work at step %s at time %s' % (
                    global_step_val, time.ctime()))
                self.start_time = time.time()
                self.start_step = global_step_val
            if self.finish_time == 0 and global_step_val >= self.end_at_global_step:
                log_fn('Finishing real work at step %s at time %s' % (
                    global_step_val, time.ctime()))
                self.finish_time = time.time()
                self.finish_step = global_step_val

    def done(self):
        return self.finish_time > 0

    def steps_per_second(self):
        return ((self.finish_step - self.start_step) /
                (self.finish_time - self.start_time))
