import scipy as sp
import scipy.stats as st
import scipy.linalg as la
import pdb
import limix
import time
from limix.core.gp import GP2KronSumLR
from limix.core.covar import FreeFormCov
from read_utils import read_geno
import pandas as pd

def inv_logdet(M,calc_inv=True):
    try:
        L = la.cholesky(M).T
        if calc_inv:
            inv = la.cho_solve((L,True),sp.eye(L.shape[0]))
        logdet = 2*sp.log(L.diagonal()).sum()
    except:
        print 'chol failed'
        S, U = la.eigh(M)
        I = S>1e-9
        S = S[I]; U = U[:,I]
        if calc_inv:
            inv = sp.dot(U*S**(-1),U.T)
        logdet = sp.log(S).sum()
    if calc_inv:
        return inv, logdet
    else:
        return logdet
        
def calc_Ai_beta_s2(yKiy,Ai,FKiy,df):
    beta = sp.dot(Ai,FKiy)
    s2 = (yKiy-sp.dot(FKiy[:,0],beta[:,0]))/df
    return beta,s2

class LMMinter():

    def __init__(self,y,F,cov=None,reml=False):
        if F is None:   F = sp.ones((y.shape[0],1))
        self.y = y
        self.F = F
        self.cov = cov
        self.df = y.shape[0]-F.shape[1]
        self.reml = reml
        self._fit_null()

    def _fit_null(self):
        """ fit the null model """
        if self.cov==None:
            self.Kiy = self.y
            self.KiF = self.F
        else:
            self.Kiy = self.cov.solve(self.y)
            self.KiF = self.cov.solve(self.F)
        self.FKiy = sp.dot(self.F.T, self.Kiy)
        self.FKiF = sp.dot(self.F.T, self.KiF)
        self.yKiy = sp.dot(self.y[:,0], self.Kiy[:,0])
        # calc beta_F0 and s20
        if self.reml:
            self.A0i, self.logdetA0 = inv_logdet(self.FKiF)
        else:
            self.A0i = la.pinv(self.FKiF)
        self.beta_F0, self.s20 = calc_Ai_beta_s2(self.yKiy,self.A0i,self.FKiy,self.df)

    def process(self, G, Inter, verbose=False):
        """ LMMstep scan """
        t0 = time.time()
        k = self.F.shape[1]
        m = Inter.shape[1]
        F1KiF1 = sp.zeros((k+m, k+m))
        F1KiF1[:k,:k] = self.FKiF
        F1Kiy = sp.zeros((k+m,1))
        F1Kiy[:k,0] = self.FKiy[:,0] 
        s2 = sp.zeros(G.shape[1])
        self.beta_g = sp.zeros([m,G.shape[1]])
        if self.reml:   df = self.df-Inter.shape[1]
        else:           df = self.df
        for s in range(G.shape[1]):
            X = G[:,[s]]*Inter
            if self.cov==None:  KiX = X 
            else:               KiX = self.cov.solve(X)
            F1KiF1[k:,:k] = sp.dot(X.T,self.KiF)
            F1KiF1[:k,k:] = F1KiF1[k:,:k].T
            F1KiF1[k:,k:] = sp.dot(X.T, KiX) 
            F1Kiy[k:,0] = sp.dot(X.T,self.Kiy[:,0])
            #this can be sped up by using block matrix inversion, etc
            if self.reml:   Ai, logdetA = inv_logdet(F1KiF1)
            else:           Ai = la.pinv(F1KiF1)
            beta,s2[s] = calc_Ai_beta_s2(self.yKiy,Ai,F1Kiy,df)
            self.beta_g[:,s] = beta[k:,0]
            if self.reml:
                XX = sp.dot(X.T,X)
                XF = sp.dot(X.T,self.F)
                _M = XX-sp.dot(XF,sp.dot(self.A0i,XF.T))
                d_logFF = inv_logdet(_M, calc_inv=False)
        #dlml and pvs
        if self.reml:
            #alt
            self.lrt = -(df*sp.log(s2)-self.df*sp.log(self.s20))
            self.lrt+= -(df-self.df)*sp.log(2*sp.pi)
            self.lrt+= -(logdetA-self.logdetA0)
            #self.lrt+= -d_logFF
        else:
            self.lrt = -self.df*sp.log(s2/self.s20)
        self.pv = st.chi2(m).sf(self.lrt)

        t1 = time.time()
        if verbose:
            print 'Tested for %d variants in %.2f s' % (G.shape[1],t1-t0)

    def getPv(self):
        return self.pv

    def getBetaSNP(self):
        return self.beta_g

    def getBetaCov(self):
        return self.beta_F

    def getLRT(self):
        return self.lrt



if __name__=="__main__":

    # gen data
    k = 10
    nSNPs = 1000
    F = pd.read_csv('data/covs.fe',sep=' ',header=None).values
    y = sp.randn(F.shape[0],1)
    E = sp.randn(y.shape[0],k)/sp.sqrt(k)
    G = 1.*(sp.rand(y.shape[0], nSNPs)<0.2)

    # fits null model
    gp = GP2KronSumLR(Y=y, Cn=FreeFormCov(1), G=E, F=F, A=sp.ones((
1,1)))
    gp.covar.Cr.setCovariance(0.5*sp.ones((1,1)))
    gp.covar.Cn.setCovariance(0.5*sp.ones((1,1)))
    gp.optimize()
    print 'sg = %.2f' % gp.covar.Cr.K()[0,0]
    print 'sn = %.2f' % gp.covar.Cn.K()[0,0]
    pdb.set_trace()

    print '.. fit REML'
    lmms_reml = LMMinter(y,F,gp.covar,reml=True)
    t0 = time.time()
    lmms_reml.process(G,E)
    print 'Elapsed:', time.time()-t0
    lrt1 = lmms_reml.getLRT()
    pdb.set_trace()

    print '.. fit noREML'
    lmms_nore = LMMinter(y,F,gp.covar)
    t0 = time.time()
    lmms_nore.process(G,E)
    print 'Elapsed:', time.time()-t0
    lrt0 = lmms_nore.getLRT()
    pdb.set_trace()

    import pylab as pl
    pl.ion()
    pl.plot(lrt1,lrt0,'.k')

    pl.ion()
    pl.subplot(221)
    pl.title('Null')
    qqplot(pv0)
    pl.subplot(222)
    pl.title('Alt')
    qqplot(pv1)
    pl.subplot(223)
    pl.title('Inter')
    qqplot(pv)
    pdb.set_trace()



