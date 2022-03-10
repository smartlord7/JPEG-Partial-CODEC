import numpy as np


def down_sample(cb, cr, variant, f):
    """
                                    Function to down sample.
                                    :param cb: CB channel.
                                    :param cr: CR channel.
                                    :param variant: downsampling variant.
                                    :param f: downsampling factor.
                                    :return: the downsampled channels.
    """
    if variant == 1:
        cb_down_sampled = cb[:, 0::f]
        cr_down_sampled = cr[:, 0::f]
    elif variant == 2:
        cb_down_sampled = cb[0::f, 0::f]
        cr_down_sampled = cr[0::f, 0::f]
    else:
        return cb, cr

    return cb_down_sampled, cr_down_sampled


def up_sample(cb, cr, variant, f):
    """
                                       Function to up sample.
                                       :param cb: CB channel.
                                       :param cr: CR channel.
                                       :param variant: downsampling variant.
                                       :param f: downsampling factor.
                                       :return: the upsampled channels.
    """
    if variant == 1:
        cb_up_sampled = np.repeat(cb, f, axis=1)
        cr_up_sampled = np.repeat(cr, f, axis=1)
    elif variant == 2:
        cb_up_sampled = np.repeat(cb, f, axis=1)
        cr_up_sampled = np.repeat(cr, f, axis=1)
        cb_up_sampled = np.repeat(cb_up_sampled, f, axis=0)
        cr_up_sampled = np.repeat(cr_up_sampled, f, axis=0)
    else:
        return cb, cr

    return cb_up_sampled, cr_up_sampled
