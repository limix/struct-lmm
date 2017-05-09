import scipy as sp
import time
import pdb

def fdr_bh(pv):
    from rpy2.robjects.packages import importr
    from rpy2.robjects.vectors import FloatVector
    stats = importr('stats')
    p_adjust = stats.p_adjust(FloatVector(pv), method = 'BH')
    return SP.array(p_adjust)

class CompQuadFormLiu():
    """
    Pvalues using liu method from CompQuadFrom
    """
    def __init__(self):
        from rpy2.robjects.packages import importr
        from rpy2.robjects.vectors import Matrix
        from rpy2.robjects.numpy2ri import numpy2ri
        ro.numpy2ri.activate()
        self.cqf = importr('CompQuadForm')

    def getPv(self, Q, lambdas):
        RV = self.cqf.liu(Q, Matrix(lambdas))
        return RV

class CompQuadFormDavies():
    """
    Pvalues for min test statistics from SKAT-O
    """
    def __init__(self):
        import rpy2.robjects as ro
        from rpy2.robjects.vectors import FloatVector
        from rpy2.robjects.numpy2ri import numpy2ri
        ro.numpy2ri.activate()
        ro.r.source('./../include/Rcode/davies_final_integration.R')
        self.r_davies = ro.globalenv['SKAT_Optimal_PValue_Davies']

    def getPv(self, qminrho, MuQ, VarQ, KerQ, lam, VarRemain, Df, tau, rho_list, T):
        RV = self.r_davies(FloatVector(qminrho), MuQ, VarQ, KerQ, FloatVector(lam), VarRemain, Df, FloatVector(tau), FloatVector(rho_list), T)
        RV = [sp.array(RV[i]) for i in range(len(RV))]
        RV = RV[0].flatten()[0]
        return RV

class CompQuadFormLiuMod():
    """
    Pvs from modified Liu method in SKAT
    """
    def __init__(self):
        import rpy2.robjects as ro
        from rpy2.robjects.vectors import Matrix
        from rpy2.robjects.numpy2ri import numpy2ri
        ro.numpy2ri.activate()
        ro.r.source('./../include/Rcode/liu_mod.R')
        self.r_liu_mod = ro.globalenv['SKAT_liu.MOD']

    def getPv(self, Q, lambdas):
        RV = self.r_liu_mod(Q, Matrix(lambdas))
        return RV

class CompQuadFormDaviesSkat():
    """
    Pvs using Davies method
    """
    def __init__(self):
        import rpy2.robjects as ro
        from rpy2.robjects.vectors import Matrix
        from rpy2.robjects.numpy2ri import numpy2ri
        ro.numpy2ri.activate()
        ro.r.source('./../include/Rcode/davies_skat.R')
        self.r_davies = ro.globalenv['Get_Davies_PVal']

    def getPv(self, optimal_Q_rho, LAtAL):
        RV = self.r_davies(Q = optimal_Q_rho, K = Matrix(LAtAL))
        RV = [sp.array(RV[i]) for i in range(len(RV))]
        RV = RV[0]
        return RV

if __name__=='__main__':
    pass

