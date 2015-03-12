import cvxopt as cvx
import heapq
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize as spopt


class Archetypes():

  def __init__(self, X):

    self.X         = X
    self.n, self.p = self.X.shape

    a             = self.X.sum(axis=0) / self.n
    self.row_norm = np.dot(self.X, a)

    self.archetypesIdx  = [];
    self.archetypesList = [];

  def findArchetypes(self, m, proj='Gaussian'):

    if proj.lower() == 'gaussian':
      XS = np.dot(self.X , np.random.randn(self.p, m)) / self.row_norm[:,np.newaxis]
      self.archetypesIdx.extend( np.argmax(XS, axis=0) )
      self.archetypesIdx.extend( np.argmin(XS, axis=0) )

    else:
      raise ValueError("proj unrecognized.")

    self.archetypesList = [ (idx, self.archetypesIdx.count(idx)) for idx in set(self.archetypesIdx) ]
    self.archetypesList = sorted(self.archetypesList, key=lambda archetype: archetype[1], reverse=True)

    return self.archetypesList

  def findAllArchetypes(self, p=0.05, proj='Gaussian'):

    kk = 0
    while len(self.findArchetypes(m = np.ceil(1.5/p), proj=sampling)) > kk:
      kk = len(self.archetypesList)

    return self.archetypesList

  def screePlot(self):

    ll = [ archetype[1] for archetype in self.archetypesList ]

    screeFig = plt.figure()
    plt.plot(ll)
    plt.show()

    return screeFig

  def weights(self, k=0, loss='square'):

    if k > 0:
      self.k = k
      ll = [ archetype[0] for archetype in self.archetypesList[:k] ]

    else:
      self.k = len(self.archetypesList)
      ll = [ archetype[0] for archetype in self.archetypesList ]

    self.H = self.X[ll,:]
    self.W = np.zeros((self.n, len(ll) ))
    self.r = np.zeros(self.n)

    for i, x in enumerate(self.X):
      if i not in ll:
        if loss.lower() == 'square':
          self.W[i,:], self.r[i] = spopt.nnls(self.H.T,x.T)

        elif loss.upper() == 'KL':
            raise NotImplementedError

            """
            \nabla_H D(X||WH) = - W' * Y + W' * Z, where Y = X ./ (W * H), and z_{ij} = 1.
            \nabla_W D(X||WH) = - Y * H' + Z * H'
            """

            def F(w=None, z=None):
              if w is None:
                return 0, matrix(1.0, (self.k,1))

              if min(w) <= 0.0:
                return None

              f  = -sum(log(x))
              Df = -(x**-1).T

              if z is None:
                return f, Df

              else:
                H = spdiag(z[0] * x**-2)
                return f, Df, H

        else:
          raise ValueError("loss unrecognized.")

      else:
        self.W[i,ll.index(i)] = 1.

    return self.W, self.H


if __name__ == '__main__':

  k, n, p = 20, 1000, 2000

  H = np.random.randn(k,p)
  W = np.vstack((np.eye(k), np.random.rand(n-k,k)))
  X = np.dot(W, H)

  a              = Archetypes(X)
  archetypesList = a.findAllArchetypes()
  a.screePlot()
