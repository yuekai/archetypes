import heapq
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize as opt


class Archetypes():

  def __init__(self, X):

    self.X         = X
    self.n, self.p = self.X.shape

    a             = self.X.sum(axis=0) / self.n
    self.row_norm = np.dot(self.X, a)

    self.archetypesIdx  = [];
    self.archetypesList = [];

  def findArchetypes(self, m):

    XS = np.dot(self.X , np.random.randn(self.p, m)) / self.row_norm[:,np.newaxis]

    self.archetypesIdx.extend( np.argmax(XS, axis=0) )
    self.archetypesIdx.extend( np.argmin(XS, axis=0) )

    self.archetypesList = [ (idx, self.archetypesIdx.count(idx)) for idx in set(self.archetypesIdx) ]

    return self.archetypesList

  def findAllArchetypes(self, p=0.05):

    kk = 0
    while len(self.findArchetypes(m = np.ceil(1.5/p))) > kk:
      kk = len(self.archetypesList)

    return self.archetypesList

  def screePlot(self):

    l = [ self.archetypesIdx.count(idx) for idx in set(self.archetypesIdx) ]
    l.sort(reverse=True)

    screeFig = plt.figure()
    plt.plot(l)
    plt.show()

    return screeFig

  def weights(self, k=0):

    if k > 0:
      l  = heapq.nlargest(k, [ archetype[1] for archetype in self.archetypesList ])
      ll = [ archetype[0] for archetype in self.archetypesList if archetype[1] >= l[-1]]
    else:
      ll = [ archetype[0] for archetype in self.archetypesList ]

    self.H = self.X[ll,:]
    self.W = np.zeros((self.n, len(ll) ))
    self.r = np.zeros(self.n)

    for i, x in enumerate(self.X):
      if i in ll:
        self.W[i,ll.index(i)] = 1.
      else:
        self.W[i,:], self.r[i] = opt.nnls(self.H.T,x.T)

    return self.H, self.W


if __name__ == '__main__':

  k, n, p = 20, 1000, 2000

  H = np.random.randn(k,p)
  W = np.vstack((np.eye(k), np.random.rand(n-k,k)))
  X = np.dot(W, H)

  a              = Archetypes(X)
  archetypesList = a.findAllArchetypes()
  a.screePlot()
