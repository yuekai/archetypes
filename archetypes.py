import cvxopt as cvx
from cvxopt import solvers as cvxsol
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize as spopt

cvxsol.options['show_progress'] = False


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
    while len(self.findArchetypes(m=np.ceil(1.5/p), proj=proj)) > kk:
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
    self.W = np.zeros((self.n, self.k ))
    self.r = np.zeros(self.n)

    for i, x in enumerate(self.X):
      if i not in ll:
        if loss.lower() == 'square':
          self.W[i,:], self.r[i] = spopt.nnls(self.H.T, x.T)

        elif loss.upper() == 'KL':
            
            HH = cvx.matrix(self.H.T)
            ek = cvx.matrix(1., (self.k,1))
            ep = cvx.matrix(1., (self.p,1))

            def KL(ww=None, zz=None):
              if ww is None:
                return 0, cvx.matrix(1., (self.k,1))

              if cvx.min(ww) <= 0.:
                return None

              xx   = cvx.matrix(x.T)
              HHww = HH * ww

              kl   = sum(cvx.mul(xx, cvx.log(cvx.div(xx, HHww)) + HHww - xx ))
              Dkl  = (ep.T - cvx.div(xx.T, HHww.T)) * HH

              if zz is None:
                return kl, Dkl

              else:
                D2kl = HH.T * cvx.spdiag(cvx.div(zz[0] * xx, HHww**2)) * HH
                return kl, Dkl, D2kl

            II        = cvx.matrix(0., (self.k, self.k))
            II[::self.k+1] = 1.  
            
            sol = cvxsol.cp(KL, G=-II, h=ek)
            self.W[i,:] = np.array(sol['x']).squeeze()

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
