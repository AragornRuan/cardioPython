# -*- coding: utf-8 -*-

import numpy as np
from scipy import signal
from pywt import wavedec, waverec
import math
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

class CutST(object):

#lead为x导联，向量
    def __init__(self, lead):
        self.__lead = lead
        self.__sfrep = 500

    def cut(self):
        #smooth window when sfreq = 500
        WW, p = 100, 8
        #threshold for recognizing R wave
        thresd = 0.65
        leadLen = self.__lead.size

        max_h = np.max(self.__lead[round(leadLen/4):round(3*leadLen/4)])
        rExistArea = self.__lead >= thresd * max_h
        rLeft = np.diff(np.hstack([0, rExistArea*1])) == 1
        rRight = np.diff(np.hstack([rExistArea*1, 0])) == -1
        left = np.nonzero(rLeft)
        left = left[0]
        right = np.nonzero(rRight)
        right = right[0]

        maxloc = np.zeros(len(left))
        minloc = np.zeros(len(left))
        for i in range(len(left)):
            first = left[i]
            last = right[i] + 1
            maxTemp = np.argmax(self.__lead[first:last])
            minTemp = np.argmin(self.__lead[first:last])
            maxloc[i] = maxTemp + first
            minloc[i] = minTemp + first

        #求出了R点的坐标集
        rIndex = maxloc[maxloc<leadLen-self.__sfrep]
        rIndex = rIndex[rIndex > self.__sfrep]
        # rIndex = maxloc
        # rr = []
        # for i in range(len(rIndex)):
        #     if (rIndex[i] < leadLen - self.__sfrep) and (rIndex[i] > self.__sfrep):
        #         rr.append(rIndex[i])
        # rIndex = rr

        #求出Ka，Kb坐标
        rr = len(rIndex)
        j = 0
        ka = np.zeros(rr-1)
        kb = np.zeros(rr-1)
        #每两个R波尖峰的差值
        rIntval = np.zeros(rr-1)
        for i in range(1, len(rIndex)):
            rInt = rIndex[i] -  rIndex[i-1]
            rIntval[i-1] = rInt
            if rInt < math.ceil(220 * self.__sfrep / 250):
                ka[i-1] = rIndex[i-1] + math.floor(0.5*math.sqrt(rInt)+20*self.__sfrep/250)
                kb[i-1] = rIndex[i-1] + math.floor(0.15*math.sqrt(rInt)+30*self.__sfrep/250)
            else:
                ka[i-1] = rIndex[i-1] + math.floor(0.5*math.sqrt(rInt)+25*self.__sfrep/250)
                kb[i-1] = rIndex[i-1] + 80*self.__sfrep/250

        #求T波的起始点令其为J点
        rLen = len(ka)
        s = np.zeros(rLen)
        r = 6
        for i in range(rLen):
            aK = np.zeros(int(kb[i]-ka[i]+1))
            for k in range(int(ka[i]), int(kb[i]+1)):
                skBar = np.sum(self.__lead[(k-p):(k+p+1)]) / (2*p+1)
                aK[i] = np.sum(self.__lead[k:(k+WW)] - skBar)
            maxVaule = np.max(aK)
            maxIndex = np.argmax(aK)
            minValue = np.min(aK)
            minIndex = np.argmin(aK)

            if ((1/r) < abs(maxVaule)/abs(minValue)) and (abs(maxVaule)/abs(minValue) < r):
                s[i] = max(maxIndex, minIndex) + ka[i]
            elif abs(minValue) > abs(maxVaule):
                s[i] = minIndex + ka[i]
            else:
                s[i] = maxIndex + ka[i]

        #求出T波终点即K
        LJ = len(s)
        K = np.zeros(LJ)
        for i in range(LJ):
            K[i] = s[i] + math.floor(180*self.__sfrep/1000)
            while self.__lead[int(K[i])] > 0.005:
                K[i] = K[i] + 1
        #求出ST的初始点
        J = np.zeros(LJ)
        for i in range(LJ):
            HR = math.floor(60*self.__sfrep/rIntval[i])
            if HR < 100:
                J[i] = rIndex[i] + math.floor(0.6*s[i]-rIndex[i])
            elif HR < 110:
                J[i] = rIndex[i] + math.floor(0.55*s[i]-rIndex[i])
            elif HR < 120:
                J[i] = rIndex[i] + math.floor(0.5*s[i]-rIndex[i])
            else:
                J[i] = rIndex[i] + math.floor(0.45*s[i]-rIndex[i])

        #若LJ大于25，取中间25个数据，若小于25保持原值
        if LJ > 25:
            mid = math.floor(LJ / 2)
            J = J[(mid-11):(mid+14)]
            K = K[(mid-11):(mid+14)]
            LJ = 25

        #将拼接处附近的不连贯ST-T段剔除
        c2del = 0
        lenSubSfreq = leadLen / 2 - self.__sfrep
        lenAddSfreq = leadLen / 2 + self.__sfrep
        for i in range(LJ):
            if (J[i] > lenSubSfreq) and (J[i] < lenAddSfreq) or (K[i] > lenSubSfreq) and (K[i] < lenAddSfreq):
                J[i] = 0
                K[i] = 0
                c2del = c2del + 1
        LJ = LJ - c2del
        J = J[J != 0]
        K = K[K != 0]

        return J,K,LJ

class MergeSTT(object):

    def __init__(self, vcg, J, K, sttCount):
        self.__vcg = vcg
        self.__J = J
        self.__K = K
        self.__sttCount = sttCount

    def merge(self):
        nCols = 0
        for i in range(self.__sttCount):
            nCols = nCols + self.__K[i] - self.__J[i] + 1
        stt = np.zeros((3, int(nCols)))
        currentBlockStart, currentBlockEnd = 0, 0
        for i in range(self.__sttCount):
            currentBlockEnd = currentBlockStart + self.__K[i] - self.__J[i]
            stt[:, int(currentBlockStart):int(currentBlockEnd)] = self.__vcg[:, int(self.__J[i]):int(self.__K[i])]
            currentBlockStart = currentBlockEnd + 1
        return stt

class Learn(object):
    #stt是一个n行3列的矩阵，3列表示三个导联
    def __init__(self, stt):
        self.__sfreq = 500
        self.__stt = stt
        self.__width = 0.07
        self.__u = 3
        self.__norLev = 0.9
        self.__eta = 0.8 * self.__width

    def learn(self):
        nRows = self.__stt.shape[0]
        numCent = int(np.ceil(2 / self.__width))
        M = int(pow(numCent, 3))
        cent = np.zeros((M, 3))
        A = np.ones(3)
        ijk = np.zeros(3)
        for i in range(1, numCent+1):
            for j in range(1, numCent+1):
                for k in range(1, numCent+1):
                    xRow = int((i - 1) * pow(numCent, 2) + (j - 1) * numCent + k - 1)
                    ijk[0] = i
                    ijk[1] = j
                    ijk[2] = k
                    cent[xRow,:] = A - (ijk - 1) * self.__width
        xNorm = np.zeros(nRows)
        for i in range(nRows):
            xNorm[i] = np.linalg.norm(self.__stt[i,:])
        sttMax = np.max(xNorm)
        self.__stt = (self.__stt / sttMax) * self.__norLev
        S = np.zeros((M, nRows))
        for i in range(nRows):
            xyz1 = np.zeros(3)
            xyz2 = np.zeros(3)
            mpq = np.zeros(3)
            s = np.zeros((M, 1))
            for m in range(-self.__u, self.__u):
                for p in range(-self.__u, self.__u):
                    for q in range(-self.__u, self.__u):
                        xyz1 = np.floor((A-self.__stt[i])/self.__width) + A * 0.5
                        mpq[0],mpq[1],mpq[2] = m,p,q
                        xyz2 = xyz1 + mpq
                        xyz2[xyz2 < 1] = 1
                        xyz2[xyz2 > numCent] = numCent
                        mRow = int((xyz2[0] - 1) * pow(numCent, 2) + (xyz2[1] - 1) * numCent + xyz2[2] - 1)
                        s[mRow] = math.exp(-pow(np.linalg.norm(self.__stt[i]-cent[mRow]), 2) / pow(self.__eta, 2))
            S[:,i] = s[:,0]

        p2p = np.zeros(nRows)
        for i in range(nRows - 1):
            p2p[i] = np.linalg.norm(self.__stt[i+1] - self.__stt[i])
        p2p[nRows-1] = np.linalg.norm(self.__stt[nRows-1] - self.__stt[0])
        p2pSort = np.sort(p2p)[::-1]
        p2p_30 = p2pSort[29]

        W11 = np.zeros((M, 1))
        W21 = np.zeros((M, 1))
        W31 = np.zeros((M, 1))
        W12 = np.zeros((M, 1))
        W22 = np.zeros((M, 1))
        W32 = np.zeros((M, 1))
        W13 = np.zeros((M, 1))
        W23 = np.zeros((M, 1))
        W33 = np.zeros((M, 1))
        xHat = np.zeros((3,3))

        lamb = 10
        alpha = 1.99
        a = 0.99
        TS = 1 / self.__sfreq
        repeat = 70
        nRowsRepeat = nRows * repeat

        for i in range(1, nRowsRepeat-1):
            ii = i % nRows
            iif = ii - 1
            if ii == 0:
                ii = nRows - 1
                iif = ii -1
            if ii == 1:
                iif  = nRows - 1
            if (np.linalg.norm(self.__stt[ii] - self.__stt[iif]) <= p2p_30):
                den = 1 + np.asscalar(lamb * np.transpose(S[:,iif]) * S[:,iif])
                W12 = W11 - alpha * lamb * (xHat[1,0] - self.__stt[ii,0] - a * (xHat[0,0] - self.__stt[iif,0])) * S[:,iif] / den
                W22 = W21 - alpha * lamb * (xHat[1,1] - self.__stt[ii,1] - a * (xHat[0,1] - self.__stt[iif,1])) * S[:,iif] / den
                W32 = W31 - alpha * lamb * (xHat[1,2] - self.__stt[ii,2] - a * (xHat[0,2] - self.__stt[iif,2])) * S[:,iif] / den
            xHat[2,0] = self.__stt[ii,0] + a * (xHat[1,0] - self.__stt[ii,0]) + np.asscalar(TS * np.transpose(W12) * S[:,ii])
            xHat[2,1] = self.__stt[ii,1] + a * (xHat[1,1] - self.__stt[ii,1]) + np.asscalar(TS * np.transpose(W22) * S[:,ii])
            xHat[2,2] = self.__stt[ii,2] + a * (xHat[1,2] - self.__stt[ii,2]) + np.asscalar(TS * np.transpose(W32) * S[:,ii])
            xHat[0] = xHat[1]
            xHat[1] = xHat[2]
            xHat[2] = 0

            if i >= nRowsRepeat - 301:
                W13 = W12 + W13
                W23 = W22 + W23
                W33 = W32 + W33
            W = np.zeros((3, M))
            W[0] = np.transpose(W13)
            W[1] = np.transpose(W23)
            W[2] = np.transpose(W33)

            WS = W * S / 300
            return WS


if __name__ == '__main__':
    ecg = np.genfromtxt('ecg.txt')
    ecg = ecg.transpose()
    pretreat = Pretreat(ecg)
    ecgAndVcg = pretreat.process()
    cutST = CutST(ecgAndVcg[12])
    J, K, LJ = cutST.cut()
    mergeSTT = MergeSTT(ecgAndVcg[12:15], J, K, LJ)
    stt = mergeSTT.merge()
    learn = Learn(stt.transpose())
    WS = learn.learn()

    print(WS)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(WS[0],WS[1],WS[2])
    plt.show()





