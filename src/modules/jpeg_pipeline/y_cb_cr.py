from modules.util import float_to_uint8


def rgb_to_y_cb_cr(rgb, y_cb_cr_matrix):
    """
                Converts RGB to YCBCR.
                :param rgb: RGB matrix.
                :param y_cb_cr_matrix: YCBCR default values matrix.
                :return: YCBCR converted matrix.
    """
    y_cb_cr = rgb.dot(y_cb_cr_matrix.T)
    y_cb_cr[:, :, [1, 2]] += 128

    return y_cb_cr


def y_cb_cr_to_rgb(y_cb_cr, y_cb_cr_inverse_matrix):
    """
                    Converts RGB to YCBCR.
                    :param y_cb_cr: YCBCR matrix.
                    :param y_cb_cr_inverse_matrix: YCBCR inverse default values matrix.
                    :return: RGB converted matrix.
    """
    y_cb_cr[:, :, [1, 2]] -= 128
    rgb = y_cb_cr.dot(y_cb_cr_inverse_matrix.T)
    rgb = float_to_uint8(rgb)

    return rgb
