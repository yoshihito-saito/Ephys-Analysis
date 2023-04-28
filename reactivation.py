def AssemblyWeight(Qref):
    '''
        This function returns normarized weight vectors of the assembly patterns
        Parameters
        ----------
        Qref: binned spike trains during reference epoch
    '''    
    from sklearn.decomposition import FastICA
    
    Qref = stats.zscore(Qref, axis=1)
    nCells = len(Qref)
    
    def pcacov(C):
        lambdas, PCs = np.linalg.eigh(C)
        for i in range(len(PCs[0])):
            if PCs[:,i][np.argmax(abs(PCs[:,i]))] < 0:
                PCs[:,i] = PCs[:,i]*-1
        PCs = np.fliplr(PCs)
        PCs = np.round(PCs, 4)
        return lambdas, PCs

    Cref = np.corrcoef(Qref);
    lambdas, PCs = pcacov(Cref);

    lMax = (1 + math.sqrt(nCells / len(Qref[1])))**2
    nPCs = sum(lambdas>lMax);
    if nPCs > 0:
        phi = lambdas[np.array(np.where(lambdas>lMax)).min():np.array(np.where(lambdas>lMax)).max()+1]/lMax
        phi = phi[::-1]
        PCs = PCs[:,:nPCs]
        Zproj = Qref.T.dot(PCs)
        ica = FastICA(max_iter=500000)
        source = ica.fit_transform(Zproj) # Reconstruct signals
        icaW = ica._unmixing.T  # Get estimated unmixing matrix
        Vec=(PCs.dot(icaW)) #weight vectors of the assembly patterns
        nW=Vec/(np.sqrt(sum(Vec**2))) #normalized by unit length
        for i in range(len(nW[0])):
            if nW[:,i][np.argmax(abs(nW[:,i]))] < 0:
                nW[:,i] = nW[:,i]*-1
    else:
        print('no significant PCs')
        nW=np.zeros([Qref.shape[0],1])
    return nW


def AssemblyStrength(nW, Qtar):
    '''
        This function returns assembly strength
        Parameters
        ----------
        nW: normilized weight vector
        Qtar: binned spike trains during target epoch
    '''    

    Qtar = stats.zscore(Qtar, axis=1)
    react = []
    for k in range(len(nW[0])):
        Pk = np.outer(nW[:,k], nW[:,k])
        Pk[[range(len(nW)), range(len(nW))]] = 0
        Rk = Qtar.T.dot(Pk)*Qtar.T
        Rk = np.sum(Rk, axis=1)
        react.append(Rk)
    react = np.array(react).T
    
    return react

