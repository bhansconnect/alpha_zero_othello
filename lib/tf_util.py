import tensorflow as tf
import keras.backend as K


def update_memory(fraction):
    tfconfig = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            per_process_gpu_memory_fraction=fraction,
            allow_growth=True,
        )
    )
    sess = tf.Session(config=tfconfig)
    K.set_session(sess)