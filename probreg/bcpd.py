from __future__ import print_function
from __future__ import division
import abc
from collections import namedtuple
import six
import numpy as np
import scipy as sp
import open3d as o3
from . import transformation as tf
from . import math_utils as mu


EstepResult = namedtuple('EstepResult', ['pt1', 'p1', 'n_p', 'x_hat'])
MstepResult = namedtuple('MstepResult', ['transformation', 'u_hat', 'sigma_mat', 'alpha', 'sigma2', 'q'])


@six.add_metaclass(abc.ABCMeta)
class BayesianCoherentPointDrift():
    """Bayesian Coherent Point Drift algorithm.

    Args:
        source (numpy.ndarray, optional): Source point cloud data.
    """
    def __init__(self, source=None):
        self._source = source
        self._tf_type = None
        self._callbacks = []

    def set_source(self, source):
        self._source = source

    def set_callbacks(self, callbacks):
        self._callbacks.extend(callbacks)

    @abc.abstractmethod
    def _initialize(self, target):
        return MstepResult(None, None, None)

    def expectation_step(self, t_source, target, scale, alpha, sigma_mat, sigma2, w=0.0):
        """Expectation step for CPD
        """
        assert t_source.ndim == 2 and target.ndim == 2, "source and target must have 2 dimensions."
        dim = t_source.shape[1]
        pmat = np.zeros((t_source.shape[0], target.shape[0]))
        for i in range(t_source.shape[0]):
            pmat[i, :] = np.sum(np.square(target - np.tile(t_source[i, :], (target.shape[0], 1))),
                                axis=1)
        pmat = np.exp(-pmat / (2.0 * sigma2))
        pmat /= (2.0 * np.pi * sigma2) ** (dim * 0.5)
        pmat = pmat.T
        pmat *= np.exp(-scale**2 / (2 * sigma2) * np.diag(sigma_mat) * dim)
        pmat *= (1.0 - w) * alpha
        den = w / target.shape[0] + (1.0 - w) * np.sum(pmat * alpha, axis=1)
        den[den==0] = np.finfo(np.float32).eps
        pmat = np.divide(pmat.T, den)

        pt1 = np.sum(pmat, axis=0)
        p1  = np.sum(pmat, axis=1)
        dnu_inv = 1.0 / np.kron(p1, np.ones(dim))
        px = np.dot(np.kron(pmat, np.identity(dim)), target)
        x_hat = np.multiply(px, dnu_inv)
        return EstepResult(pt1, p1, np.sum(p1), x_hat)

    def maximization_step(self, target, estep_res, sigma2_p=None):
        return self._maximization_step(self._source, target, estep_res, sigma2_p)

    @staticmethod
    @abc.abstractmethod
    def _maximization_step(source, target, estep_res, sigma2_p=None):
        return None

    def registration(self, target, w=0.0,
                     maxiter=50, tol=0.001):
        assert not self._tf_type is None, "transformation type is None."
        res = self._initialize(target)
        q = res.q
        for _ in range(maxiter):
            t_source = res.transformation.transform(self._source)
            estep_res = self.expectation_step(t_source, target, res.sigma2, w)
            res = self.maximization_step(target, estep_res, res.sigma2)
            for c in self._callbacks:
                c(res.transformation)
            if abs(res.q - q) < tol:
                break
            q = res.q
        return res


class CombinedBCPD(BayesianCoherentPointDrift):
    def __init__(self, source=None, update_scale=True, update_nonrigid_term=True):
        super(CombinedBCPD, self).__init__(source)
        self._tf_type = tf.CombinedTransformation
        self._update_scale = update_scale
        self._update_nonrigid_term = update_nonrigid_term

    def _initialize(self, target):
        dim = self._source.shape[1]
        self.gmat = mu.inverse_multiquadric_kernel(self._source, self._source)
        sigma2 = mu.squared_kernel_sum(self._source, target)
        q = 1.0 + target.shape[0] * ndim * 0.5 * np.log(sigma2)
        return MstepResult(self._tf_type(np.identity(ndim), np.zeros(ndim)), sigma2, q)

    def maximization_step(self, target, estep_res, sigma2_p=None):
        return self._maximization_step(self._source, target, estep_res,
                                       sigma2_p, self._update_scale)

    @staticmethod
    def _maximization_step(source, target, estep_res,
                           sigma2_p=None, update_scale=True):
        pt1, p1, n_p, x_hat = estep_res
        dim = source.shape[1]
        sigma_mat_inv = lmd * self.gmat + scale**2 / (sigma2_p**2) * np.diat(p1)
        return
