# -*- coding: utf-8 -*-
# Adapted SKAT-O script including covariates but is generalised for cases when no covariates exist
# Does not include centering options

import scipy as sp
import scipy.linalg as la
import scipy.stats as st
from limix_core.covar import FreeFormCov
from limix_core.gp import GP2KronSumLR  
import pdb
from .pvmixchi2 import *

def P(gp, M):
    RV = gp.covar.solve(M)
    if gp.mean.F is not None:
        WKiM = sp.dot(gp.mean.W.T, RV)
        WAiWKiM = sp.dot(gp.mean.W, gp.Areml.solve(WKiM))
        KiWAiWKiM = gp.covar.solve(WAiWKiM)
        RV-= KiWAiWKiM
    return RV


class StructLMM():

    def __init__(self, y, Env, K=None, W =None, rho_list = None):
        self.y = y
        self.Env = Env
        # K is kernel under the null (exclusing noise)
        self.K = K
        # W is low rank verion of kernel
        self.W = W
        self.rho_list = rho_list
        if self.rho_list == None:
            self.rho_list = sp.arange(0,1.01, 0.2)
        # This is to correct when Sigma is a block of ones as choice of rho parameter makes no difference to p-value but final p-value is inaccurate.  A similar correction is made in the original SKAT-O script
        #if self.rho_list[-1] == 1:
        #    self.rho_list[-1] = 0.999
        self.vec_ones = sp.ones((1, self.y.shape[0]))


    def fit_null(self, F = None, verbose=True):
        # F is a fixed effect covariate matrix with dim = N by D
        # F itself cannot have any cols of 0's and it won't work if it is None
        self.F = F
        self.qweliumod = CompQuadFormLiuMod()
        self.qwedavies = CompQuadFormDavies()
        self.qwedaviesskat =  CompQuadFormDaviesSkat()
        if self.K is not None:
            # Decompose K into low rank version
            S_K, U_K = la.eigh(self.K)
            S = sp.array([i for i in S_K if i>1e-9])
            U = U_K[:, -len(S):]
            # In most cases W = E but have left it as seperate parameter for flexibility
            self.W = U*S**0.5
            self.gp = GP2KronSumLR(Y = self.y, F = self.F, A = sp.eye(1), Cn = FreeFormCov(1), G = self.W)
            self.gp.covar.Cr.setCovariance(0.5*sp.ones((1,1)))
            self.gp.covar.Cn.setCovariance(0.5*sp.ones((1,1)))
            RV = self.gp.optimize(verbose=verbose)
            # Get optimal kernel parameters
            self.covarparam0 = self.gp.covar.Cr.K()[0,0]
            self.covarparam1 = self.gp.covar.Cn.K()[0,0]
            self.Kiy = self.gp.Kiy()
        elif self.W is not None:
            self.gp = GP2KronSumLR(Y = self.y, F = self.F, A = sp.eye(1), Cn = FreeFormCov(1), G = self.W)
            self.gp.covar.Cr.setCovariance(0.5*sp.ones((1,1)))
            self.gp.covar.Cn.setCovariance(0.5*sp.ones((1,1)))
            RV = self.gp.optimize(verbose=verbose)
            self.covarparam0 = self.gp.covar.Cr.K()[0,0]#getParams()[0]
            self.covarparam1 = self.gp.covar.Cn.K()[0,0]
            self.Kiy = self.gp.Kiy()
        else:
            # If there is no kernel then solve analytically
            self.alpha_hat = sp.dot(sp.dot(la.inv(sp.dot(self.F.T, self.F)), self.F.T), self.y)
            yminus_falpha_hat = self.y - sp.dot(self.F, self.alpha_hat)
            self.covarparam1 = (yminus_falpha_hat**2).sum()/yminus_falpha_hat.shape[0]
            self.covarparam0 = 0   
            self.Kiy = (1/float(self.covarparam1))*self.y
            self.W = sp.zeros(self.y.shape)
            RV  = self.covarparam0
        return RV

    def score_2_dof(self, X, snp_dim='col', debug=False):
        #1. calculate Qs and pvs
        Q_rho = sp.zeros(len(self.rho_list))
        Py = P(self.gp, self.y)
        xoPy = X*Py
        for i in xrange(len(self.rho_list)):
            rho = self.rho_list[i]
            LT = sp.vstack((rho**0.5*self.vec_ones, (1-rho)**0.5*self.Env.T))
            LTxoPy = sp.dot(LT, X*Py)
            Q_rho[i] = 0.5*sp.dot(LTxoPy.T, LTxoPy)

        # Calculating pvs is split into 2 steps
        # If we only consider one value of rho i.e. equivalent to SKAT and used for interaction test
        if len(self.rho_list) == 1:
            rho = self.rho_list[0]
            L = sp.hstack((rho**0.5*self.vec_ones.T, (1-rho)**0.5*self.Env))
            xoL = X*L
            PxoL = P(self.gp, xoL)
            LToxPxoL = 0.5*sp.dot(xoL.T, PxoL)
            pval = self.qwedaviesskat.getPv(Q_rho[0], LToxPxoL)
            # Script ends here for interaction test
            return pval, self.rho_list
        #or if we consider multiple values of rho i.e. equivalent to SKAT-O and used for association test
        else:
            pliumod = sp.zeros((len(self.rho_list), 4))
            for i in xrange(len(self.rho_list)):
                rho = self.rho_list[i]
                L = sp.hstack((rho**0.5*self.vec_ones.T, (1-rho)**0.5*self.Env))
                xoL = X*L
                PxoL = P(self.gp, xoL)
                LToxPxoL = 0.5*sp.dot(xoL.T, PxoL)
                eighQ, UQ = la.eigh(LToxPxoL)
                pliumod[i, ] = self.qweliumod.getPv(Q_rho[i], eighQ)
            T = pliumod[:, 0].min()
            rho_opt = pliumod[:, 0].argmin()
            optimal_rho = self.rho_list[rho_opt]
            #if optimal_rho == 0.999:
            #    optimal_rho = 1

            # 2. Calculate qmin
            qmin = sp.zeros(len(self.rho_list))
            percentile = 1-T
            for i in xrange(len(self.rho_list)):
                q = st.chi2.ppf(percentile, pliumod[i, 3])
                # Recalculate p-value for each Q rho of seeing values at least as extreme as q again using the modified matching moments method
                qmin[i] = (q - pliumod[i, 3])/(2*pliumod[i, 3])**0.5 *pliumod[i, 2] + pliumod[i, 1]


            # 3. Calculate quantites that occur in null distribution
            Px1 = P(self.gp, X)
            m = 0.5*sp.dot(X.T, Px1)
            xoE = X*self.Env
            PxoE = P(self.gp, xoE)
            ETxPxE = 0.5*sp.dot(xoE.T, PxoE)
            ETxPx1 = sp.dot(xoE.T, Px1)
            ETxPx11xPxE = 0.25/m*sp.dot(ETxPx1, ETxPx1.T)
            ZTIminusMZ = ETxPxE-ETxPx11xPxE
            eigh, vecs = la.eigh(ZTIminusMZ)

            eta = sp.dot(ETxPx11xPxE, ZTIminusMZ)
            vareta = 4*sp.trace(eta)

            OneZTZE = 0.5*sp.dot(X.T, PxoE)
            tau_top = sp.dot(OneZTZE, OneZTZE.T)
            tau_rho = sp.zeros(len(self.rho_list))
            for i in xrange(len(self.rho_list)):
                tau_rho[i] = self.rho_list[i]*m+(1-self.rho_list[i])/m*tau_top

            MuQ = sp.sum(eigh)
            VarQ = sp.sum(eigh**2)*2 + vareta
            KerQ = sp.sum(eigh**4)/(sp.sum(eigh**2)**2) * 12
            Df = 12/KerQ

            #4. Integration
            pvalue = self.qwedavies.getPv(qmin, MuQ, VarQ, KerQ, eigh, vareta, Df, tau_rho, self.rho_list, T)

            # Final correction to make sure that the p-value returned is sensible
            multi = 3
            if len(self.rho_list)<3:
                multi = 2
            idx = sp.where(pliumod[:, 0]>0)[0]
            pval = pliumod[:, 0].min()*multi 
            if pvalue<=0 or len(idx)<len(self.rho_list):
                pvalue = pval
            if pvalue==0:
                if len(idx) >0:
                    pvalue = pliumod[:, 0][idx].min()     

            if debug:
                info = {'Qs': Q_rho, 'pvs_liu': pliumod, 'qmin': qmin,
                        'MuQ': MuQ, 'VarQ': VarQ,
                        'KerQ': KerQ, 'lambd': eigh,
                        'VarXi': vareta, 'Df': Df, 'tau': tau_rho}
                return pvalue, optimal_rho, info
            else:
                return pvalue, optimal_rho




 
