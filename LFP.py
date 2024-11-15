class lfpAnalysis:
    def __init__(self, session_info):
        self.session_info = session_info
        #load unit feature
        LFP_list = []
        for i in session_info.index:
            LFP_dir = glob.glob(session_info['sessionDir'][i]+'/06_LFP_output')
            if LFP_dir != []:
                a = LFP_dir
                LFP_list.append(a)
        self.LFP_list=LFP_list

        LFP_mouseID=[]
        for i in range(len(LFP_list)):
            target = '/TM'
            idx = LFP_list[i][0].find(target)
            a = LFP_list[i][0][idx+1:idx+6]
            LFP_mouseID.append(a)
        self.LFP_mouseID = LFP_mouseID

        LFP_condition=[]        
        for i in LFP_mouseID:
            a = list(session_info.filter(items=[i],axis=0)['condition'])
            LFP_condition.append(a)
        LFP_condition = sum(LFP_condition,[])
        self.LFP_condition = LFP_condition   
        
        self.sf=30000
        
    def load_LFP(self, mouseID):
        LFPDir = glob.glob(self.session_info['sessionDir'][mouseID]+'/06_LFP_output')[0]
        BLA_LFP=np.load(LFPDir+'/BLA_LFP.npy')
        M2_LFP=np.load(LFPDir+'/M2_LFP.npy')
        S1_LFP=np.load(LFPDir+'/S1_LFP.npy')

        
        #load TTL timing
        ttlDir = glob.glob(self.session_info['sessionDir'][mouseID]+'/00_TTL_output/*.h5')
        ttlh5 = h5py.File(ttlDir[0],'r')
        epoch_TTL_timing = ttlh5['epoch_TTL_timing'].value
        pre_epoch_id = list([0,epoch_TTL_timing[1]-epoch_TTL_timing[0]])
        task_epoch_id = list([epoch_TTL_timing[1]-epoch_TTL_timing[0], epoch_TTL_timing[1]-epoch_TTL_timing[0]+epoch_TTL_timing[3]-epoch_TTL_timing[2]])
        post_epoch_id = list([epoch_TTL_timing[1]-epoch_TTL_timing[0]+epoch_TTL_timing[3]-epoch_TTL_timing[2],epoch_TTL_timing[1]-epoch_TTL_timing[0]+epoch_TTL_timing[3]-epoch_TTL_timing[2]+epoch_TTL_timing[5]-epoch_TTL_timing[4]])   
        ttlh5.flush()
        ttlh5.close()
        epoch_id = [pre_epoch_id, task_epoch_id, post_epoch_id]
        
        #load sleep data
        stateDir = glob.glob(self.session_info['sessionDir'][mouseID]+'/05_sleepstate_output/*.h5')
        stateh5 = h5py.File(stateDir[0], 'r')
        sleep_state = stateh5['sleep_state'].value
        stateh5.flush()
        stateh5.close()
        LFPs = {'BLA':BLA_LFP, 'M2':M2_LFP, 'S1':S1_LFP, 'State': sleep_state, 'Epoch': epoch_id}
        del BLA_LFP, M2_LFP, S1_LFP, sleep_state, epoch_id
        
        return LFPs
    
    def HFO_detect(self, LFP, state, low, high, th_low, th_high, plot=True):
        '''
            This function detects high frequency oscillations.
            Parameters
            ----------
            low: band pass frequency low
            high: band pass frequency high
            sf: sampling frequency
            th_low: lower threshold 
            th_high: higher threshold 
        '''
        def bandpass(data, low, high, sf, order):
            '''
                bandpass filter
                Parameters
                ----------
                low: bandpass low
                high: bandpas high
                sf: sampling frequency
            '''
            nyq = sf/2
            low = low/nyq
            high = high/nyq
            b, a = butter(order, [low, high], btype='band')
            filtered_data = filtfilt(b, a, data)
            return filtered_data
        
        def binupsampling(signal_d, binwidth_wide, binwidth):
            signal_up = np.zeros(int(len(signal_d)*(binwidth_wide/binwidth)))
            for i,k in enumerate(signal_d):
                signal_up[int(i*(binwidth_wide/binwidth)):int((i+1)*(binwidth_wide/binwidth))] = k
            return signal_up

        #down sampling
        sf = self.sf
        if state.shape[0]%(sf*0.025) != 0:
            state = state[:-int(state.shape[0]%int(0.025*sf))]
            LFP = LFP[:-int(LFP.shape[0]%int(0.025*sf))]
        state_d_25ms = state[::int(sf*0.025)]
        state_d = binupsampling(state_d_25ms,  0.025, 0.001)
        
        convert_rate=1000 #resampling rate
        LFP_d = LFP[::int(sf/convert_rate)]
        LFP_d = LFP_d[np.where(state_d==1)[0]] #NREM
        #filter
        LFP_HFO = bandpass(LFP_d, low, high, convert_rate, order=2)

        #hilbert transformation
        hil_LFP_HFO = abs(signal.hilbert(LFP_HFO))
        #convert to z-score
        z_HFO = stats.zscore(hil_LFP_HFO)

        cand_HFO = np.where(z_HFO>=th_low)[0]

        #sort serial indices
        cand_HFO_sort = []
        tmp = [cand_HFO[0]]
        for i in range(len(cand_HFO)-1):
            if cand_HFO[i+1] - cand_HFO[i] == 1:
                tmp.append(cand_HFO[i+1])
            else:
                if len(tmp) > 0:
                    cand_HFO_sort.append(tmp)
                tmp = []
                tmp.append(cand_HFO[i+1])

        #combine candidates within 20ms
        for i in range(1,len(cand_HFO_sort)):
            if (min(cand_HFO_sort[i])-max(cand_HFO_sort[i-1]))<convert_rate*0.02:
                a = sum([cand_HFO_sort[i-1],cand_HFO_sort[i]],[])
                cand_HFO_sort[i-1]=None
                cand_HFO_sort[i]=a
        cand_HFO_sort_2 = list(filter(None, cand_HFO_sort))

        #remove candidates under 30ms duration
        for i in range(len(cand_HFO_sort_2)):
            if len(cand_HFO_sort_2[i])<convert_rate*0.03:
                cand_HFO_sort_2[i]=None
        cand_HFO_sort_3 = list(filter(None, cand_HFO_sort_2))

        #remove candidates with peaks <th_high z-score
        for i in range(len(cand_HFO_sort_3)):
            if len(np.array(np.where(z_HFO[cand_HFO_sort_3[i]]>=th_high)).reshape(-1))==0:
                cand_HFO_sort_3[i]=None
        HFO_idx = list(filter(None, cand_HFO_sort_3))

        col_list=['start_idx', 'peak_idx', 'end_idx', 'duration']
        HFOs = pd.DataFrame(columns=col_list)
        # HFO_train = np.zeros(len(LFP))
        binwidth=0.025
        HFO_train = np.zeros(len(LFP[::int(sf*binwidth)]))
        for i in range(len(HFO_idx)):
            peak = max(z_HFO[HFO_idx[i]])
            peak_idx_org = np.array((HFO_idx[i][z_HFO[HFO_idx[i]].tolist().index(peak)]*(sf/convert_rate)), dtype='int')
            strat_idx_org = np.array((min(HFO_idx[i])*(sf/convert_rate)), dtype='int')
            end_idx_org = np.array((max(HFO_idx[i])*(sf/convert_rate)), dtype='int')
            duration = (end_idx_org-strat_idx_org)/sf*1000 #msec
            a = pd.Series([strat_idx_org, peak_idx_org, end_idx_org, duration], index=HFOs.columns)
            HFOs = HFOs.append(a, ignore_index=True)
            HFO_train[round(strat_idx_org/sf/binwidth)]=1
            
        if plot==True:
            a=np.ones(len(HFOs))
            time=np.linspace(0, len(LFP_d)/convert_rate,len(LFP_d))

            upper=figure(height=200, width=1200)
            upper.line(time[0:30000], LFP_d[0:30000])
            upper.x_range=Range1d(start=0, end=5)
            upper.y_range=Range1d(start=-1000, end=1000)

            bottom=figure(height=200, width=1200)
            bottom.line(time[0:30000], z_HFO[0:30000], color='OrangeRed')
            bottom.circle(HFOs['start_idx'].values/(sf), a*8, color='blue')
            bottom.circle(HFOs['peak_idx'].values/(sf), a*8, color='red')
            bottom.circle(HFOs['end_idx'].values/(sf), a*8, color='green')
            bottom.x_range=upper.x_range
            bottom.y_range=Range1d(start=-3, end=12)       

            p=gridplot([[upper],[bottom]])
            show(p)

        return HFOs, HFO_train

    def SW_detect(self, LFP, state):

        def highpass(data, freq, sf, order):
            '''
                bandpass filter
                Parameters
                ----------
                low: bandpass low
                high: bandpas high
                sf: sampling frequency
            '''
            from scipy.signal import butter, filtfilt
            nyq = sf/2
            freq = freq/nyq
            b, a = butter(order, freq, btype='high')
            filtered_data = filtfilt(b, a, data)

            return filtered_data

        def lowpass(data, freq, sf, order):
            '''
                bandpass filter
                Parameters
                ----------
                low: bandpass low
                high: bandpas high
                sf: sampling frequency
            '''
            from scipy.signal import butter, filtfilt
            nyq = sf/2
            freq = freq/nyq
            b, a = butter(order, freq, btype='low')
            filtered_data = filtfilt(b, a, data)
            return filtered_data

        def pos2negZerocrossings(x):
            pos = x > 0
            npos = ~pos
            return (pos[:-1] & npos[1:]).nonzero()[0]

        def neg2posZerocrossings(x):
            pos = x > 0
            npos = ~pos
            return (npos[:-1] & pos[1:]).nonzero()[0]

        def find_indices(peak_idx, peak_sorted, pos2negZerocrossings_idx):
            d = []
            for i in range(0,len(pos2negZerocrossings_idx)+1):
                tmp = list(peak_idx[np.where(peak_sorted==i)[0]])
                d.append(tmp)
            return d

        def delete_duplicated_peaks(LFP_SW, peak_idx, peak_sorted_idx):
            peak_idx_2 = []
            for i in range(len(peak_sorted_idx)):
                if len(peak_sorted_idx[i])>1:
                    peak_idx_2.append(peak_sorted_idx[i][list(LFP_SW[peak_sorted_idx[i]]).index(max(LFP_SW[peak_sorted_idx[i]]))])
                elif len(peak_sorted_idx[i])==1:
                    peak_idx_2.append(peak_sorted_idx[i][0])

            return np.array(peak_idx_2)
        
        def phase_detect(slow_wave):
            down_phase = np.array([slow_wave['start_idx'].values.tolist(), slow_wave['mid_crossing_idx'].values.tolist()], dtype=np.int).T
            up_phase = np.array([slow_wave['mid_crossing_idx'].values.tolist(), slow_wave['end_idx'].values.tolist()], dtype=np.int).T
            return np.array([down_phase, up_phase])
        
        def phase_train(SW_phase_train, phase):
            down_phase=phase[0]
            up_phase=phase[1]
            for i in range(len(down_phase)):
                SW_phase_train[down_phase[i][0]:down_phase[i][1]]=1
            for i in range(len(up_phase)):
                SW_phase_train[up_phase[i][0]:up_phase[i][1]]=-1

            return SW_phase_train
        
        def binupsampling(signal_d, binwidth_wide, binwidth):
            signal_up = np.zeros(int(len(signal_d)*(binwidth_wide/binwidth)))
            for i,k in enumerate(signal_d):
                signal_up[int(i*(binwidth_wide/binwidth)):int((i+1)*(binwidth_wide/binwidth))] = k
            return signal_up
        
        sf = self.sf 
        if state.shape[0]%(sf*0.025) != 0:
            state = state[:-int(state.shape[0]%int(0.025*sf))]
            LFP = LFP[:-int(LFP.shape[0]%int(0.025*sf))]
        state_d_25ms = state[::int(sf*0.025)]
        state_d = binupsampling(state_d_25ms,  0.025, 0.001)
        
        convert_rate=1000 #resampling rate
        LFP_d = LFP[::int(sf/convert_rate)]
        LFP_d = LFP_d[np.where(state_d==1)[0]] #NREM
        
        LFP_tmp = highpass(LFP_d, freq=0.1, sf=convert_rate, order=2)
        LFP_SW = lowpass(LFP_tmp, freq=4, sf=convert_rate, order=5)

        pos2negZerocrossings_idx = pos2negZerocrossings(LFP_SW)
        neg2posZerocrossings_idx = neg2posZerocrossings(LFP_SW)
        pos_peak_idx = find_peaks(LFP_SW)[0]
        neg_peak_idx = find_peaks(-LFP_SW)[0]

        pos_peak_sorted = np.searchsorted(pos2negZerocrossings_idx, pos_peak_idx, side='left')
        neg_peak_sorted = np.searchsorted(pos2negZerocrossings_idx, neg_peak_idx, side='left')
        pos_peak_sorted_idx = find_indices(pos_peak_idx, pos_peak_sorted, pos2negZerocrossings_idx)
        neg_peak_sorted_idx = find_indices(neg_peak_idx, neg_peak_sorted, pos2negZerocrossings_idx)


        pos_peak_idx_2 = delete_duplicated_peaks(LFP_SW, pos_peak_idx, pos_peak_sorted_idx)
        neg_peak_idx_2 = delete_duplicated_peaks(-LFP_SW, neg_peak_idx, neg_peak_sorted_idx)

        wave_idx = np.array([neg2posZerocrossings_idx[i:i+2] for i in range(len(neg2posZerocrossings_idx)-1)])
        pos2negZerocrossings_idx = np.delete(pos2negZerocrossings_idx, np.where(pos2negZerocrossings_idx<np.min(wave_idx))[0])
        pos2negZerocrossings_idx = np.delete(pos2negZerocrossings_idx, np.where(pos2negZerocrossings_idx>np.max(wave_idx))[0])

        pos_peak_idx_3 = np.delete(pos_peak_idx_2, np.where(pos_peak_idx_2<np.min(neg2posZerocrossings_idx))[0])
        pos_peak_idx_4 = np.delete(pos_peak_idx_3, np.where(pos_peak_idx_3>np.max(pos2negZerocrossings_idx))[0])
        neg_peak_idx_3 = np.delete(neg_peak_idx_2, np.where((neg_peak_idx_2<np.min(pos2negZerocrossings_idx))|(neg_peak_idx_2<np.min(neg2posZerocrossings_idx)))[0])
        neg_peak_idx_4 = np.delete(neg_peak_idx_3, np.where(neg_peak_idx_3>np.max(neg2posZerocrossings_idx))[0])

        wave_idx_org = np.array(wave_idx/convert_rate*sf, dtype='int')
        wave_all = pd.DataFrame(wave_idx_org, columns=['start_idx', 'end_idx'])
        wave_all['mid_crossing_idx'] = np.array(pos2negZerocrossings_idx/convert_rate*sf, dtype='int')
        wave_all['peak_idx'] = np.array(pos_peak_idx_4/convert_rate*sf, dtype='int')
        wave_all['trough_idx'] = np.array(neg_peak_idx_4/convert_rate*sf, dtype='int')
        wave_all['duration (ms)'] = (wave_all['end_idx']-wave_all['start_idx'])/sf*1000
        wave_all['peak_to_trough (ms)'] = (wave_all['trough_idx']-wave_all['peak_idx'])/sf*1000
        wave_all['peak_to_mid (ms)'] = (wave_all['mid_crossing_idx']-wave_all['peak_idx'])/sf*1000
        wave_all['mid_to_trough (ms)'] = (wave_all['trough_idx']-wave_all['mid_crossing_idx'])/sf*1000
        wave_all['peak_amp'] = LFP_SW[pos_peak_idx_4]
        wave_all['trough_amp'] = LFP_SW[neg_peak_idx_4]

        wave_all['down_state'] = wave_all['peak_amp'] > wave_all['peak_amp'].quantile(0.85)
        wave_all['up_state'] = wave_all['trough_amp'] < wave_all['trough_amp'].quantile(0.40)

        wave_all['slow_oscillation'] = (wave_all['down_state']==True) & (wave_all['up_state']==True) & (wave_all['peak_to_trough (ms)']>100) & (wave_all['peak_to_trough (ms)']<500)
        wave_all['delta_oscillation'] = (wave_all['down_state']==False) & (wave_all['up_state']==True) & (wave_all['peak_to_trough (ms)']<500)
        slow_wave = wave_all[(wave_all['slow_oscillation']==True) | (wave_all['delta_oscillation']==True)]
        
        
        # donw sampling
        binwidth = 0.025
        delta_peak_train = np.zeros(len(LFP[::int(sf*binwidth)]))
        SO_peak_train = np.zeros(len(LFP[::int(sf*binwidth)]))
        
        delta_peak_train[np.round(slow_wave[slow_wave['delta_oscillation']]['peak_idx']/sf/binwidth).astype('int')] = 1 #hyperpolized peak idx
        delta_peak_train[np.round(slow_wave[slow_wave['delta_oscillation']]['trough_idx']/sf/binwidth).astype('int')] = -1 #upstate peak idx
        SO_peak_train[np.round(slow_wave[slow_wave['slow_oscillation']]['peak_idx']/sf/binwidth).astype('int')] = 1 #downstate peak idx
        SO_peak_train[np.round(slow_wave[slow_wave['slow_oscillation']]['trough_idx']/sf/binwidth).astype('int')] = -1 #upstate peak idx           
        
        delta_phase_train = np.zeros(len(LFP))
        SO_phase_train = np.zeros(len(LFP))
        delta_phase = phase_detect(slow_wave[slow_wave['delta_oscillation']])
        delta_phase_train = phase_train(delta_phase_train, delta_phase)
        SO_phase = phase_detect(slow_wave[slow_wave['slow_oscillation']])
        SO_phase_train = phase_train(SO_phase_train, SO_phase)
        
        
        delta_phase_train = delta_phase_train[::int(sf*binwidth)]
        SO_phase_train = SO_phase_train[::int(sf*binwidth)]
        
        slow_wave_train = {'delta_peak':delta_peak_train, 'so_peak':SO_peak_train, 'delta_phase':delta_phase_train, 'so_phase':SO_phase_train}
        
        return slow_wave, slow_wave_train