# -*- coding: utf-8 -*-

import numpy as np
from scipy import signal
from pywt import wavedec, waverec
import math

class WaveletFilter(object):

    def __init__(self, vector, wavename, level):
        self.__vector = vector
        self.__wavename = wavename
        self.__level = level

    def __ddencmp(self, vector):

        #Set threshold default value.
        n = vector.size
        thr = math.sqrt(2 * math.log(n))

        dwt_output = wavedec(vector, 'db1', level = 1)
        details = sorted(abs(dwt_output[1]))
        normaliz = 0.0

        if len(details) % 2 !=  0:
            normaliz = details[int((len(details)-1)/2)]
        else:
            normaliz = (details[int((len(details)-2)/2)] + details[int(len(details)/2)]) / 2

        return thr * normaliz  / 0.6745

    def __wdencmp(self, dwt_output, thr):


        temp = abs(dwt_output) - thr
        temp = (temp + abs(temp)) / 2

        for i in range(1, len(dwt_output)):
            ay = dwt_output[i]
            dwt_output[i][ay>0] = temp[i][ay>0]
            dwt_output[i][ay<0] = -temp[i][ay<0]

        return waverec(list(dwt_output), self.__wavename)

    def filter(self):
        dwt_output = wavedec(self.__vector, self.__wavename, level=self.__level)
        thr = self.__ddencmp(self.__vector)
        return self.__wdencmp(np.array(dwt_output), thr)

class MedianFilter(object):

    def __init__(self, vector, window):
        self.__vector = vector
        self.__window = window

    def filter(self):
        n_exten = self.__window / 2 if self.__window % 2 == 0 else (self.__window - 1) / 2
        exten = np.zeros([int(n_exten)])
        vector = np.hstack([exten, self.__vector, exten])

        result = []
        for i in range(0, len(self.__vector)):
            result.append(np.median(vector[i:int(i+self.__window)]))

        return np.array(result)



class Pretreat(object):

    #ECG格式为每个导联一行，一共有12行
    def __init__(self, ecg):
        self.__ecg = ecg

    def process(self):
        sfreq, gain = 500, 250
        ecg = self.__ecg[:, 2000-sfreq:12000+sfreq]
        vcg_x = 0.38*ecg[0] - 0.07*ecg[1] - 0.13*ecg[6] + 0.05*ecg[7] - 0.01*ecg[8] + 0.14*ecg[9] + 0.06*ecg[10] + 0.54*ecg[11]
        vcg_y = -0.17*ecg[0] + 0.93*ecg[1] + 0.06*ecg[6] - 0.02*ecg[7] - 0.05*ecg[8] - 0.17*ecg[9] + 0.06*ecg[10] + 0.13*ecg[11]
        vcg_z = 0.11*ecg[0] + 0.23*ecg[1] - 0.43*ecg[6] -0.06*ecg[7] - 0.14*ecg[8] - 0.2*ecg[9] - 0.11*ecg[10] + 0.31*ecg[11]

        ecg_vcg = np.vstack([ecg, vcg_x, vcg_y, vcg_z])
        ecg_vcg = ecg_vcg / gain
        
        medResult = []
        for e in ecg_vcg:
            m = signal.medfilt(e, int(200 * sfreq / 1000 + 1))
            m = signal.medfilt(m, int(600 * sfreq / 1000 + 1))
            medResult.append(m)
        ecg_vcg = ecg_vcg - np.array(medResult)

        result = []
        for e in ecg_vcg:
            wf = WaveletFilter(e, 'coif4', 7)
            wfResult = wf.filter()
            result.append(wfResult)

        return np.array(result)
