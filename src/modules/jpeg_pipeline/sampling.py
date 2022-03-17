import numpy as np


def parse_down_sample_variant(variant):
    variant = variant.split(":")
    y_fac = int(variant[0])
    cb_fac = int(variant[1])
    cr_fac = int(variant[2])
    f = int(y_fac / cb_fac)

    return cb_fac, cr_fac, f


def down_sample(cb, cr, variant):
    """
                                    Function to down sample.
                                    :param cb: CB channel.
                                    :param cr: CR channel.
                                    :param variant: downsampling variant.
                                    :param f: downsampling factor.
                                    :return: the downsampled channels.
    """

    cb_fac, cr_fac, f = parse_down_sample_variant(variant)

    if f == 1:
        return cb, cr

    if cb_fac == cr_fac:
        cb_down_sampled = cb[:, 0::f]
        cr_down_sampled = cr[:, 0::f]
    elif cr_fac == 0:
        cb_down_sampled = cb[0::f, 0::f]
        cr_down_sampled = cr[0::f, 0::f]
    else:
        return cb, cr

    return cb_down_sampled, cr_down_sampled


def up_sample(cb, cr, variant):
    """
                                       Function to up sample.
                                       :param cb: CB channel.
                                       :param cr: CR channel.
                                       :param variant: downsampling variant.
                                       :param f: downsampling factor.
                                       :return: the upsampled channels.
    """

    variant = variant.split(":")
    y_fac = int(variant[0])
    cb_fac = int(variant[1])
    cr_fac = int(variant[2])
    f = y_fac / cb_fac

    if f == 1:
        return cb, cr

    if cb_fac == cr_fac:
        cb_up_sampled = np.repeat(cb, f, axis=1)
        cr_up_sampled = np.repeat(cr, f, axis=1)
    elif cr_fac == 0:
        cb_up_sampled = np.repeat(cb, f, axis=1)
        cr_up_sampled = np.repeat(cr, f, axis=1)
        cb_up_sampled = np.repeat(cb_up_sampled, f, axis=0)
        cr_up_sampled = np.repeat(cr_up_sampled, f, axis=0)
    else:
        return cb, cr

    return cb_up_sampled, cr_up_sampled
