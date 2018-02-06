import numpy as np
import scipy
import os
import sys
from numpy import linalg as LA
from time import *
import codecs, commands, threading

def chrompwr(X, P=.5):
    nchr, nbts = X.shape
    # norms of each input col
    CMn = np.tile(np.sqrt(np.sum(X * X, axis=0)), (nchr, 1))
    CMn[np.where(CMn==0)] = 1
    # normalize each input col, raise to power
    CMp = np.power(X/CMn, P)
    # norms of each resulant column
    CMpn = np.tile(np.sqrt(np.sum(CMp * CMp, axis=0)), (nchr, 1))
    CMpn[np.where(CMpn==0)] = 1.
    # rescale cols so norm of output cols match norms of input cols
    return CMn * (CMp / CMpn)

def fftshift(X):
    """
    Same as fftshift in Matlab
    -> python: TBM, 2011-11-05, TESTED
    ok, useless, but needed to be sure it was doing the same thing
    """
    # return scipy.fftpack.fftshift(X)
    return np.fft.fftshift(X)

def magnitude(X):
    """
    Magnitude of a complex matrix
    """
    r = np.real(X)
    i = np.imag(X)
    return np.sqrt(r * r + i * i);

def fft2(X):
    """
    Same as fft2 in Matlab
    -> python: TBM, 2011-11-05, TESTED
    ok, useless, but needed to be sure it was doing the same thing
    """
    # return scipy.fftpack.fft2(X)
    return np.fft.fft2(X)

def btchroma_to_fftmat(btchroma, win=75):

    nchrm, nbeats = btchroma.shape
    assert nchrm == 12, 'beat-aligned matrix transposed?'
    if nbeats < win:
        return None
    # output
    fftmat = np.zeros((nchrm * win, nbeats - win + 1))
    for i in range(nbeats-win+1):
        patch = fftshift(magnitude(fft2(btchroma[:,i:i+win])))
        # 'F' to copy Matlab, otherwise 'C'
        fftmat[:, i] = patch.flatten('F')
    return fftmat

def chromnorm(F, P=2.):
    """
    N = chromnorm(F,P)
       Normalize each column of a chroma ftrvec to unit norm
       so cross-correlation will give cosine distance
       S returns the per-column original norms, for reconstruction
       P is optional exponent for the norm, default 2.
    2006-07-14 dpwe@ee.columbia.edu
    -> python: TBM, 2011-11-05, TESTED
    """
    nchr, nbts = F.shape
    if not np.isinf(P):
        S = np.power(np.sum(np.power(F,P), axis=0),(1./P));
    else:
        S = F.max()
    S = S if S > 0 else 1
    # try:
    #     d = F/S
    # except:
    #     print F, S
    return F / S 

# bt_chroma
def extract_feats(feats, win=75):
    PWR = 1.96
    WIN = win

    # apply pwr
    feats = chrompwr(feats, PWR)
    # extract fft
    feats = btchroma_to_fftmat(feats, WIN)
    if feats is None:
        return None
    # return the non-normalized features (L, 900)
    return feats.T

def extract_2DFM(feats, win=75):
    # 1.- Beat Synchronous Chroma
    # 2.- L2-Norm
    # 3.- Shingle (PATCH_LEN: 75 x 12)
    # 4.- 2D-FFT
    # 5.- L2-Norm
    # 6.- Log-Scale
    # 7.- Sparse Coding
    # 8.- Shrinkage
    #. 9.- Median Aggregation

    # 11.- L2-Norm
    H = extract_feats(feats, win)
    if H is None:
        return None
    H = np.median(H, axis=0)
    two_dfm = chromnorm(H.reshape(H.shape[0], 1)).squeeze()
    return two_dfm.astype('float32')

in_path, out_path = sys.argv[1], sys.argv[2]
chroma = np.load(in_path)
np.save(out_path, extract_2DFM(chroma.T))
