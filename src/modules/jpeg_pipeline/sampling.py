"""------------DESTRUCTIVE COMPRESSION OF IMAGE - PARTIAL JPEG CODEC------------
University of Coimbra
Degree in Computer Science and Engineering
Multimedia
3rd year, 2nd semester
Authors:
Rui Bernardo Lopes Rodrigues, 2019217573, uc2019217573@student.uc.pt
Sancho Amaral Simões, 2019217590, uc2019217590@student.uc.pt
Tiago Filipe Santa Ventura, 2019243695, uc2019243695@student.uc.pt
Coimbra, 23rd March 2022
---------------------------------------------------------------------------"""


import cv2
import numpy as np


# region Public Methods

def parse_down_sample_variant(variant: str):
    """
    Function that parses the down sample variant.
    :param variant: the down-sample variant.
    :return: the Down sampled cb and cr factor.
    """
    variant = variant.split(":")
    y_fac = int(variant[0])
    cb_fac = int(variant[1])
    cr_fac = int(variant[2])
    f = int(y_fac / cb_fac)

    return cb_fac, cr_fac, f


def down_sample(cb: np.ndarray, cr: np.ndarray, variant: str, interpolation_type: int = None):
    """
    Function that down samples Cb and Cr channels.
    :param cb: the CB channel.
    :param cr: the CR channel.
    :param variant: the down sampling variant to use (e.g. 4:2:2, 4:2:0, 4:4:4, ...).
    :param interpolation_type: the chosen interpolation (e.g. LINEAR; CUBIC; AREA).
    :return: the down-sampled channels.
    """

    cb_fac, cr_fac, f = parse_down_sample_variant(variant)
    n_rows = cb.shape[0]
    n_cols = cb.shape[1]

    if f == 1:
        return cb, cr

    if cb_fac == cr_fac:
        if interpolation_type is not None:
            cb_down_sampled = cv2.resize(cb, (int(n_cols / f), n_rows), interpolation=interpolation_type)
            cr_down_sampled = cv2.resize(cr, (int(n_cols / f), n_rows), interpolation=interpolation_type)
        else:
            cb_down_sampled = cb[:, 0::f]
            cr_down_sampled = cr[:, 0::f]

    elif cr_fac == 0:
        if interpolation_type is not None:
            cb_down_sampled = cv2.resize(cb, (int(n_cols / f), int(n_rows / f)), interpolation=interpolation_type)
            cr_down_sampled = cv2.resize(cr, (int(n_cols / f), int(n_rows / f)), interpolation=interpolation_type)
        else:
            cb_down_sampled = cb[0::f, 0::f]
            cr_down_sampled = cr[0::f, 0::f]
    else:
        return cb, cr

    return cb_down_sampled, cr_down_sampled


def up_sample(cb: np.ndarray, cr: np.ndarray, variant: str, interpolation_type: int = None):
    """
    Function that up-samples Cb and Cr channels.
    :param cb: the Cb channel.
    :param cr: the Cr channel.
    :param variant: the down-sampling variant used (e.g. 4:2:2, 4:2:0, 4:4:4, ...).
    :param interpolation_type: the chosen interpolation (e.g. LINEAR; CUBIC; AREA).
    :return: the up-sampled channels.
    """

    variant = variant.split(":")
    y_fac = int(variant[0])
    cb_fac = int(variant[1])
    cr_fac = int(variant[2])
    f = int(y_fac / cb_fac)

    if f == 1:
        return cb, cr

    n_rows = cb.shape[0]
    n_cols = cb.shape[1]

    if cb_fac == cr_fac:
        if interpolation_type is not None:
            cb_up_sampled = cv2.resize(cb, (n_cols * f, n_rows), interpolation=interpolation_type)
            cr_up_sampled = cv2.resize(cr, (n_cols * f, n_rows), interpolation=interpolation_type)
        else:
            cb_up_sampled = np.repeat(cb, f, axis=1)
            cr_up_sampled = np.repeat(cr, f, axis=1)
    elif cr_fac == 0:
        if interpolation_type is not None:
            cb_up_sampled = cv2.resize(cb, (n_cols * f, n_rows * f), interpolation=interpolation_type)
            cr_up_sampled = cv2.resize(cr, (n_cols * f, n_rows * f), interpolation=interpolation_type)
        else:
            cb_up_sampled = np.repeat(cb, f, axis=1)
            cr_up_sampled = np.repeat(cr, f, axis=1)
            cb_up_sampled = np.repeat(cb_up_sampled, f, axis=0)
            cr_up_sampled = np.repeat(cr_up_sampled, f, axis=0)
    else:
        return cb, cr

    return cb_up_sampled, cr_up_sampled

# endregion Public Methods
