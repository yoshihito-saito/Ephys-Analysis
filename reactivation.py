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


def prepareReactTrial(predictor, predicted, React_Event, binwidth, binwidth_fine):
    '''
        Create trial matrix for GLM
        Parameters
        ----------
        predictor: binned raster of predictor units (1 ms bin)
        predicted: binned raster of predicted units (1 ms bin)
        React_Event: reactivation event index (25 ms bin)
        binwidth: binwidth for reactivation analysis, 25 ms 
        binwidth_fine: binwidth for binned raster, 1 ms  
    '''    
    # convert to fine sampling rate 
    React_Event = (React_Event*binwidth/binwidth_fine).astype('int64')
    
    # delete React_Event index outside window
    del_id_pre = np.where((React_Event-binwidth_fine)<=0)
    React_Event = np.delete(React_Event, del_id_pre)

    del_id_post = np.where((React_Event+binwidth_fine)>predictor.shape[1])
    React_Event = np.delete(React_Event, del_id_post)

    # number of React_Event
    nReact = React_Event.shape[0] 

    react_window_idx = np.vstack([React_Event, React_Event+(binwidth/binwidth_fine)]).T.astype('int64')
    predictor_mtx = []
    predicted_mtx = []
    for i in range(nReact):
        a = np.sum(predictor[:,react_window_idx[i][0]:react_window_idx[i][1]],axis=1)
        b = np.sum(predicted[:,react_window_idx[i][0]:react_window_idx[i][1]],axis=1)
        predictor_mtx.append(a)
        predicted_mtx.append(b)
    predictor_mtx = np.array(predictor_mtx)
    predicted_mtx = np.array(predicted_mtx)
    
    #constant = 1 
    predictor_mtx = np.hstack((predictor_mtx, np.ones(predictor_mtx.shape[0]).reshape(predictor_mtx.shape[0],1)))
    
    ## Create Shuffled Data
    predicted_mtx_shuffled=sklearn.utils.shuffle(predicted_mtx)

    TrialMatrix = {'predictor': predictor_mtx, 'predicted': predicted_mtx, 'predicted_shuffled': predicted_mtx_shuffled}

    return TrialMatrix


def spikeGLMReact(TrialMatrix):
    '''
        GLM for predicting the firing of one neuron from ensemble of other neurons
        Parameters
        ----------
        TrialMatrix
    '''  
    from sklearn.model_selection import KFold
    import statsmodels.api as sm
    
    nTrials = np.arange(TrialMatrix['predictor'].shape[0])
    kf = KFold(n_splits = 5, shuffle = True) # 5folds
    nCells = TrialMatrix['predicted'].shape[1]

    prediction_gain = []
    prediction_gain_shfl = []
    for k in range(nCells):
        prediction_gain_tmp2=[]
        prediction_gain_shfl_tmp2=[]
        for i in range(100): # repeat 100 times
            prediction_gain_tmp = []
            prediction_gain_shfl_tmp = []
            for train_index, test_index in kf.split(nTrials):
                y_trn = TrialMatrix['predicted'][train_index]
                y_test = TrialMatrix['predicted'][test_index]

                x_trn = TrialMatrix['predictor'][train_index]
                x_test = TrialMatrix['predictor'][test_index]

                y_trn_shfl =  TrialMatrix['predicted_shuffled'][train_index]
                y_test_shfl = TrialMatrix['predicted_shuffled'][test_index]

                # create an instance of the GLM class
                if np.where(y_trn[:,k]>0)[0]!=[]:
                    link = sm.genmod.families.links.log #link function
                    
                    glm = sm.GLM(endog=y_trn[:,k], exog=x_trn, family=sm.families.Poisson(link=link))
                    result = glm.fit()

                    glm_shfl = sm.GLM(endog=y_trn_shfl[:,k], exog=x_trn, family=sm.families.Poisson(link=link))
                    result_shfl = glm_shfl.fit()

                    # predict using fitted model on the test data
                    yhat = result.predict(x_test)
                    yhat_shfl = result_shfl.predict(x_test)

                    #calculate prediction error
                    diff = np.mean(abs(y_test[:,k]/np.max(y_test[:,k])-yhat))
                    diff_sfhl = np.mean(abs(y_test_shfl[:,k]/np.max(y_test_shfl[:,k])-yhat_shfl))
                    
                    #calculate shuffled prediction error
                    diff_shuff = np.mean(abs(y_test[:,k]/np.max(y_test[:,k])-np.random.permutation(yhat)))
                    diff_shuff_sfhl = np.mean(abs(y_test_shfl[:,k]/np.max(y_test_shfl[:,k])-np.random.permutation(yhat_shfl)))
                    
                    #calculate prediction gain
                    prediction_gain_raw = diff_shuff/diff
                    prediction_gain_tmp.append(prediction_gain_raw)
                    prediction_gain_raw_shfl = diff_shuff_sfhl/diff_sfhl
                    prediction_gain_shfl_tmp.append(prediction_gain_raw_shfl)
                else:
                    prediction_gain_tmp = [np.nan]
                    prediction_gain_shfl_tmp = [np.nan]

            prediction_gain_tmp = np.nanmean(prediction_gain_tmp)
            prediction_gain_shfl_tmp = np.nanmean(prediction_gain_shfl_tmp)
            prediction_gain_tmp2.append(prediction_gain_tmp)
            prediction_gain_shfl_tmp2.append(prediction_gain_shfl_tmp)
        prediction_gain.append(np.nanmean(prediction_gain_tmp2))
        prediction_gain_shfl.append(np.nanmean(prediction_gain_shfl_tmp2))

    prediction_gain = np.array(prediction_gain)
    prediction_gain_shfl  = np.array(prediction_gain_shfl)

    return prediction_gain, prediction_gain_shfl