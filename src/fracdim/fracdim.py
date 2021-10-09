#!/usr/bin/env python

import numpy as np
from scipy.spatial import distance


def _qreg(x, y):
    """
    Optimised (hardcoded) quadratic regression.

    Parameters
    ----------
    x, y : array_like
        The independent and dependent variables for the regression.

    Returns
    -------
    intercept: float
    lp: float
    qp: float
        The final regression is the best fit for

        .. math:: y = qp \cdot x^2 + lp \cdot x + intercept
    """
    n = len(x)
    Ex = x.sum()
    Ex2 = (x*x).sum()
    Ez = Ex2
    Ez2 = (x*x*x*x).sum()
    Exz = (x*x*x).sum()
    Ey = y.sum()
    Exy = (x*y).sum()
    Ezy = (x*x*y).sum()
    det = n*(Ex2*Ez2 - Exz**2) - Ex*(Ex*Ez2 - Exz*Ez) + Ez*(Ex*Exz - Ex2*Ez)
    # equivalent
    #det = (-1)*Ex*(Ex*Ez2 - Exz*Ex) + Ex2*(n*Ez2 - Ez**2) - Exz(n*Exz - ExEz)
    #det = Ez*(Ex*Exz - Ex2*Ez) - Exz*(n*Exz - Ex*Ez) + Ez2*(n*Ex2 - Ex**2)
    w0 = Ey*(Ex2*Ez2 - Exz**2) - Exy*(Ex*Ez2 - Exz*Ez) + Ezy*(Ex*Exz - Ex2*Ez)
    w0 = w0/det
    w1 = Ey*(Ex*Ez2 - Exz*Ez) - Exy*(n*Ez2 - Ez**2) + Ezy*(n*Exz - Ex*Ez)
    w1 = (-1)*w1/det
    w2 = Ey*(Ex*Exz - Ex2*Ez) - Exy*(n*Exz - Ex*Ez) + Ezy*(n*Ex2 - Ex**2)
    w2 = w2/det
    return np.array([w0, w1, w2])


def dim(X):
    dm = distance.squareform(distance.pdist(X))
    dmu = np.linspace(0, dm.max(), 1024+1)[1:]
    if dm.shape[0] <= 1024:  # do all at once
        dmd = dm[:, :, np.newaxis] < dmu[np.newaxis, np.newaxis, :]
        NE = dmd.sum(axis=0).sum(axis=0) / (dmd.shape[0]*dmd.shape[1])
    else:  # conserve memory
        NE = np.empty(dmu.shape[0])
        for i in range(dmu.shape[0]):
            dmd = dm[:, :, np.newaxis] < dmu[np.newaxis, np.newaxis, i]
            NE[i] = dmd.sum(axis=0).sum(axis=0) / (dmd.shape[0]*dmd.shape[1])
    log_E, log_NE = np.log(dmu), np.log(NE)
    #log_E, log_NE = dmu, np.log(NE)
    div_factor = 1
    div_log_E = log_E[div_factor:-div_factor]
    div_E = log_E[div_factor*2:] - log_E[:-div_factor*2]
    div_log_NE = (log_NE[div_factor*2:] - log_NE[:-div_factor*2]) /div_E
    c, b, a = _qreg(div_log_E, div_log_NE)
    centre = -b/(2*a)
    height = c - b**2 / (4*a)
    idx = np.argmax(div_log_E > centre)
    idx = np.argmax(div_log_E > centre)
    dim, dim_max = div_log_NE[abs(idx-3):idx+3].mean(), div_log_NE.max()

    offset = 32  # about the squareroot of the 1024 points we are using
    quantile = div_log_NE[abs(idx-offset):idx+offset]
    return quantile.max(), [quantile.mean(), height, div_log_NE.max()]


def lyapunov():
    pass

