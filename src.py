from numba import vectorize, guvectorize, float32, int32, jit, cuda 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import pywt
import math
from scipy import signal
from scipy.stats import entropy
from sklearn.cluster import KMeans

def img_filter(img):
    w = 5
    wVal = 1/w
    conColMatrix = np.zeros(shape=(w,1))
    conColMatrix = np.add(conColMatrix, wVal)
    conRowMatrix = np.zeros(shape=(1,w))
    conRowMatrix  = np.add(conRowMatrix, wVal)
    a = signal.convolve2d(img, conColMatrix, boundary='symm', mode='same')
    return signal.convolve2d(a, conRowMatrix, boundary='symm', mode='same')

@jit(nopython=True)
def T_or_I_of_Neutrsophic(item, minVal, maxVal):
    return (item - minVal) / (maxVal - minVal)

@jit(nopython=True)
def EnI_Pixel(item):
    return item * math.log2(item)

@jit(nopython=True)
def Alpha_I_Pixel(item,minVal, maxVal):
    return (item * minVal) / (maxVal - minVal)

@jit(nopython=True)
def Beta_I_Pixel(item,minVal, maxVal):
    return (item * minVal) / (maxVal - minVal)

def Neutrsophic_Image(item):
    lmv = img_filter(item)
    lmvMin = np.amin(lmv)
    lmvMax = np.amax(lmv)
    TI_func = np.vectorize(T_or_I_of_Neutrsophic)
    T = TI_func(lmv,lmvMin,lmvMax)
    absMatrix = np.absolute(np.array(item) - np.array(lmv))
    absMatrixMin = np.amin(absMatrix)
    absMatrixMax = np.amax(absMatrix)
    I = TI_func(absMatrix, absMatrixMin, absMatrixMax)
    F = 1 - T
    pns = np.zeros((len(T),len(T[0]),3))
    pns[:,:,0] = T
    pns[:,:,1] = I
    pns[:,:,2] = F
    return pns

def cal_alpha_beta(img, EnI):
    h = len(img)
    w = len(img[0])
    # EnI_func = np.vectorize(EnI_Pixel)
    # EnI = EnI_func(img)
    EnMin = 0.0
    EnMax = -math.log2(1/h*w)
    alphaMin = 0.01
    alphaMax = 0.1
    alpha = alphaMin + (( alphaMax - alphaMin ) * ( EnI - EnMin ) / ( EnMax - EnMin ))
    beta = 1 - alpha
    return alpha, beta

def ns(img, enI):
    T = img[:,:,0]
    I = img[:,:,1]

    rowCount = len(T)
    colCount = len(T[0])
    # T_alpha mean
    alphaValT, betaValT = cal_alpha_beta(T, enI)
    meanT = img_filter(T)
    alphaT = np.ones((rowCount,colCount))
    alphaT[I<alphaValT] = T[I<alphaValT]
    alphaT[I>=alphaValT] = meanT[I>=alphaValT]
    alphaMT = img_filter(alphaT)
   
    # I_alpha mean
    meanI = np.absolute(alphaT-alphaMT)
    meanImin = np.amin(meanI)
    meanImax = np.amax(meanI)
    Alpha_I_Pixel_func = np.vectorize(Alpha_I_Pixel)
    alphaI = Alpha_I_Pixel_func(meanI,meanImin,meanImax)

    EnhT=np.ones((rowCount, colCount))
    EnhT[alphaT<=0.5] = 2*(EnhT[alphaT<=0.5] ** 2)
    EnhT[alphaT>0.5] = 1-2*(1-EnhT[alphaT>0.5]) ** 2
    betaEnhT=np.ones((rowCount, colCount))
    betaEnhT[alphaI<betaValT]=alphaT[alphaI<betaValT]
    betaEnhT[alphaI>=betaValT]=EnhT[alphaI>=betaValT]

    betaT = img_filter(betaEnhT)
    betaI = np.absolute(betaEnhT - betaT)
    betaImin = np.amin(betaI)
    betaImax = np.amax(betaI)
    Beta_I_Pixel_func = np.vectorize(Alpha_I_Pixel)
    betaEnhI = Beta_I_Pixel_func(betaI,betaImin,betaImax)
    value, counts = np.unique(betaEnhI, return_counts=True)
    nsEntropy = entropy(counts)
    return nsEntropy, alphaValT, betaEnhT, betaEnhI

def get_X_img(img):
    enl=0.00001
    err=0.0001
    
    while True:
        nsEntropy, alpha, betaEnhT, betaEnhI  = ns(img, enl)
        val = (nsEntropy-enl)/enl
        if(val < err):
            break
        else:
            enl = nsEntropy
    x = np.zeros((len(img), len(img[0])))
    mt = img_filter(betaEnhT)
    x[betaEnhI<alpha]=betaEnhT[betaEnhI<alpha]
    x[betaEnhI>=alpha]=mt[betaEnhI>=alpha]
    return x

img = cv2.imread('C:\\Users\\mfo\\Desktop\\calismalar\\cita_tek.jpg')
luvRawImg = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
luvImg = np.array(luvRawImg,'float')
grayRawImg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
grayImg = np.array(grayRawImg,'float')
LL, (LH, HL, HH) = pywt.dwt2(grayImg,'bior2.2')
l,u,v = luvImg[:,:,0],  luvImg[:,:,1],  luvImg[:,:,2]
# Step 4
LH = cv2.resize(LH,dsize = (len(grayImg[0]), len(grayImg)),interpolation=cv2.INTER_CUBIC)
HL = cv2.resize(HL,dsize = (len(grayImg[0]), len(grayImg)),interpolation=cv2.INTER_CUBIC)
MELH = img_filter(LH)
MEHL = img_filter(HL)
# Step 5
MELH_NS = Neutrsophic_Image(MELH)
MEHL_NS = Neutrsophic_Image(MEHL)
l_NS = Neutrsophic_Image(l)
u_NS = Neutrsophic_Image(u)
v_NS = Neutrsophic_Image(v)
# Step 6,7,8,9
l_NS_X = get_X_img(l_NS)
u_NS_X = get_X_img(u_NS)
v_NS_X = get_X_img(v_NS)
MELH_NS_X = get_X_img(MELH_NS)
MEHL_NS_X = get_X_img(MEHL_NS)
nRows = len(l_NS_X)
nCols = len(l_NS_X[0])
X1 = np.zeros((nRows,nCols,5))
X1[:,:,0] = l_NS_X
X1[:,:,1] = u_NS_X
X1[:,:,2] = v_NS_X
X1[:,:,3] = MELH_NS_X
X1[:,:,4] = MEHL_NS_X
X2 = np.reshape(X1,(nRows*nCols,5))
kmeans  = KMeans(n_clusters=4, random_state=0)
clusters = kmeans.fit_predict(X2)
resultImg = clusters.reshape(nRows,nCols)
titles = ['LL', 'LH', 'HL', 'HH', 'LL', 'MELH', 'MEHL', 'HH', 'luvImg', 'resultImg' ]
fig = plt.figure(figsize=(12, 3))
for i, a in enumerate([LL, LH, HL, HH, LL, MELH, MEHL, HH, luvRawImg, resultImg]):
    ax = fig.add_subplot(3, 4, i + 1)
    ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout()
plt.show()