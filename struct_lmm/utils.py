import sys
import scipy as SP
import h5py
import pdb
import os

def fdr_bh(pv):
    from rpy2.robjects.packages import importr
    from rpy2.robjects.vectors import FloatVector
    stats = importr('stats')
    p_adjust = stats.p_adjust(FloatVector(pv), method = 'BH')
    return SP.array(p_adjust)

def makedir(dname):
    if not os.path.exists(dname):
        os.makedirs(dname)

def smartAppend(table,name,value):
    """
    helper function for apppending in a dictionary  
    """ 
    if name not in table.keys():
        table[name] = []
    table[name].append(value)

def smartConcatenate(table,name,value):
    """
    helper function for apppending in a dictionary  
    """
    if name not in table.keys():
        table[name] = SP.zeros((value.shape[0],0))
    if len(value.shape)==1:
        value = value[:,SP.newaxis]
    table[name] = SP.concatenate((table[name],value),1)

def dumpDictHdf5(RV,o):
    """ Dump a dictionary where each page is a list or an array """
    for key in RV.keys():
        o.create_dataset(name=key,data=SP.array(RV[key]),chunks=True,compression='gzip')

def smartDumpDictHdf5(RV,o):
    """ Dump a dictionary where each page is a list or an array or still a dictionary (in this case, it iterates)"""
    for key in RV.keys():
        if type(RV[key])==dict:
            g = o.create_group(key)
            smartDumpDictHdf5(RV[key],g)
        else:
            o.create_dataset(name=key,data=SP.array(RV[key]),chunks=True,compression='gzip')

def getLambda(pv):
    """
    return lambda genomic control given the pvs
    """
    rv = SP.array([SP.median(SP.log10(pv),1)/SP.log10(0.5)])
    return rv

def getRelativePosition(pos,strand,start,end):
    """ return the relative position respect to the TSS """
    if strand=='1':
        rv=float(pos-start)
    if strand=='-1':
        rv=float(end-pos)
    else:
        rv = 1.009
    return rv

def matchIDs(ID1,ID2):
    """ match ID1 and ID2 """
    idx1 = []
    idx2 = []
    for p in range(ID1.shape[0]):
        is_in = ID1[p] in ID2
        idx1.append(is_in)
        if is_in:   idx2.append(SP.where(ID2==ID1[p])[0][0])
    idx1 = SP.array(idx1)
    idx2 = SP.array(idx2)
    return idx1, idx2

def wait(sec,verbose=True):
    """ wait sec seconds """
    import time as TIME
    if verbose:
        print "wait %s s"%sec
    start = TIME.time()
    while 1:
        if TIME.time()-start>sec:   break
    pass

def pearsCorrRavel(Y1,Y2):
    """ calculated the prearson correlation between vec(Y1) and vec(Y2) """

    y1 = Y1.ravel()
    y2 = Y2.ravel()
    rv = SP.corrcoef(y1,y2)[0,1]

    return rv


def pearsCorrMean(Y1,Y2):
    """ calculated the avg prearson correlation between columns of Y1 and Y2 """

    rv = 0
    for ic in range(Y.shape[1]):
        rv += SP.corrcoef(Y1[:,ic],Y2[:,ic])[0,1]
    rv/=float(Y.shape[1])

    return rv

def getCumPos(chrom,pos):
    """
    getCumulativePosition
    """
    n_chroms = int(chrom.max())
    x = 0
    for chrom_i in range(1,n_chroms+1):
        I = chrom==chrom_i
        pos[I]+=x
        x=pos[I].max()
    return pos

def getChromBounds(chrom,posCum):
    """
    getChromBounds
    """
    n_chroms = int(chrom.max())
    chrom_bounds = []
    for chrom_i in range(2,n_chroms+1):
        I1 = chrom==chrom_i
        I0 = chrom==chrom_i-1
        _cb = 0.5*(posCum[I0].max()+posCum[I1].min())
        chrom_bounds.append(_cb)
    return chrom_bounds


def getROC(LLR,testwindow,hotwindow):
    """
    return the tps, fps and fns
    given the tested windows and the hotwindow
    """

    #1. rank LLR and test windows
    all=SP.concatenate((LLR[:,SP.newaxis],testwindow),1)
    all.view('i8,i8,i8,i8').sort(order=['f0'], axis=0)
    all=all[::-1,:]
    testwindow = all[:,1:]

    #2. check where the true has been found
    n_tests  = testwindow.shape[0]
    for test in range(n_tests):
        boola = hotwindow[0]==testwindow[test,0]
        boolb = hotwindow[1]<=testwindow[test,2]
        boolc = hotwindow[2]>=testwindow[test,1]
        bool  = boola*boolb*boolc
        if bool:    break

    #3. build tp, fp and fn vectors (comp r is # within rank k)
    tp  = SP.concatenate((SP.zeros(test),SP.ones(n_tests-test)))
    fp  = SP.array(range(1,n_tests+1),dtype=float)-tp
    #fn = SP.ones_like(tp)-tp
    tp /= tp.max()
    fp /= fp.max()

    #5. export
    out = {}
    out['tp'] = tp
    out['fp'] = fp

    return out

def getROC1(LLR,testwindow,hotwindow):
    """
    return the tps, fps and fns
    given the tested windows and the hotwindow
    """
    
    #1. sort windows according to LLR score
    idx = SP.argsort(-LLR)
    testwindow = testwindow[idx]
    
    #all=SP.concatenate((LLR[:,SP.newaxis],testwindow),1)
    #all.view('i8,i8,i8,i8').sort(order=['f0'], axis=0)
    #all=all[::-1,:]
    #testwindow = all[:,1:]
    tp = SP.zeros(LLR.shape[0])
    fp = SP.zeros(LLR.shape[0])

    #2. check where the true has been found
    ntp = 0; nfp = 0
    n_tests  = testwindow.shape[0]
    for test in range(n_tests):
        boola = hotwindow[0]==testwindow[test,0]
        boolb = hotwindow[1]<=testwindow[test,2]
        boolc = hotwindow[2]>=testwindow[test,1]
        bool  = boola*boolb*boolc
        if bool: ntp+=1
        else:    nfp+=1
        tp[test] = ntp
        fp[test] = nfp

    tp /= tp.max()
    fp /= fp.max()

    #5. export
    out = {}
    out['tp'] = tp
    out['fp'] = fp

    return out

