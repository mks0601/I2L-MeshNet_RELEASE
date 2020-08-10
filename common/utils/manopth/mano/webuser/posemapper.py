'''
Copyright 2017 Javier Romero, Dimitrios Tzionas, Michael J Black and the Max Planck Gesellschaft.  All rights reserved.
This software is provided for research purposes only.
By using this software you agree to the terms of the MANO/SMPL+H Model license here http://mano.is.tue.mpg.de/license

More information about MANO/SMPL+H is available at http://mano.is.tue.mpg.de.
For comments or questions, please email us at: mano@tue.mpg.de


About this file:
================
This file defines a wrapper for the loading functions of the MANO model.

Modules included:
- load_model:
  loads the MANO model from a given file location (i.e. a .pkl file location),
  or a dictionary object.

'''


import chumpy as ch
import numpy as np
import cv2


class Rodrigues(ch.Ch):
    dterms = 'rt'

    def compute_r(self):
        return cv2.Rodrigues(self.rt.r)[0]

    def compute_dr_wrt(self, wrt):
        if wrt is self.rt:
            return cv2.Rodrigues(self.rt.r)[1].T


def lrotmin(p):
    if isinstance(p, np.ndarray):
        p = p.ravel()[3:]
        return np.concatenate(
            [(cv2.Rodrigues(np.array(pp))[0] - np.eye(3)).ravel()
             for pp in p.reshape((-1, 3))]).ravel()
    if p.ndim != 2 or p.shape[1] != 3:
        p = p.reshape((-1, 3))
    p = p[1:]
    return ch.concatenate([(Rodrigues(pp) - ch.eye(3)).ravel()
                           for pp in p]).ravel()


def posemap(s):
    if s == 'lrotmin':
        return lrotmin
    else:
        raise Exception('Unknown posemapping: %s' % (str(s), ))
