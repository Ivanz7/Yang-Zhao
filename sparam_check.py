# -*- encoding:UTF-8 -*-  
# S-parameter quality check
# by Ivan Zhao
# ivan.y.zhao@hotmail.com
# 2020-10-20

from numpy import empty
from numpy import array
from numpy import linalg,dot,diag,array
import numpy as np

import cmath
import math
import os
import sys

numbersList=[]
freqMul = 1e6
complexType = 'MA'
Z0=50.
sp=True
f=[]
m_sToken='S'

# Read S-parameter file
name='Via_Conn_Combined.s4p'
sfile = open(name) 

ext=str.lower(name).split('.')[-1]
N=int(str.lower(name).split('.')[-1].split('s')[1].split('p')[0])

for line in sfile:
    lineList=str.lower(line).split('!')[0].split()  
    if len(lineList)>0:       
        if lineList[0] == '#':
            if 'hz' in lineList: freqMul = 1.0
            if 'khz' in lineList: freqMul = 1e3
            if 'mhz' in lineList: freqMul = 1e6
            if 'ghz' in lineList: freqMul = 1e9
            if 'ma' in lineList: complexType = 'MA'
            if 'ri' in lineList: complexType = 'RI'
            if 'db' in lineList: complexType = 'DB'
            if 'r' in lineList:
                Z0=float(lineList[lineList.index('r')+1])
            if not m_sToken.lower() in lineList:
                sp=False
        else: numbersList.extend(lineList)
sfile.close()

frequencies = len(numbersList)//(1+N*N*2)
m_d=[empty([N,N]).tolist() for fi in range(frequencies)]

# Read Data
for fi in range(frequencies):
    f.append(float(numbersList[(1+N*N*2)*fi])*freqMul)
    for r in range(N):
        for c in range(N):
            n1=float(numbersList[(1+N*N*2)*fi+1+(r*N+c)*2])
            n2=float(numbersList[(1+N*N*2)*fi+1+(r*N+c)*2+1])
            if complexType == 'RI':
               m_d[fi][r][c]=n1+1j*n2
            else:
                expangle=cmath.exp(1j*math.pi/180.*n2)
                if complexType == 'MA':
                    m_d[fi][r][c]=n1*expangle
                elif complexType == 'DB':
                    m_d[fi][r][c]=math.pow(10.,n1/20)*expangle
            if N == 2:
                m_d[fi]=array(m_d[fi]).transpose().tolist()

# Checnk Passivity
# by Calculating the largest singular value
pasiveList=[]
passiveTol=0.0001
#for i in range(frequencies):
for m in m_d:
    passive=linalg.svd(m,full_matrices=False,compute_uv=False)[0]
    pasiveList.append(passive)
#print(pasiveList)
print('\n----------------------Check Results-----------------------\n')
if max(pasiveList) > (1+passiveTol):
    pResult='Failed'
else:
    pResult='Pass'
print('Passivity check result: %s! (The maximum singular value is %f)' %(pResult,max(pasiveList)))

# Check the Causality 
# by Calculating the CQM
Vn=np.array(m_d[1:frequencies])-np.array(m_d[0:frequencies-1])

Re_Vn=Vn.real
Im_Vn=Vn.imag

Rn=Re_Vn[1:frequencies-1]*Im_Vn[0:frequencies-2] - Im_Vn[1:frequencies-1]*Re_Vn[0:frequencies-2]
Rn_P=Rn

for aa in Rn_P:
    if (aa<0).all():
        aa=0

sum_Rn_P=Rn_P[0]
abs_Rn=np.abs(Rn)
sum_Rn=abs_Rn[0]

for i in range(1,frequencies-2):
    sum_Rn_P=sum_Rn_P+Rn_P[i]
    sum_Rn=sum_Rn+abs_Rn[i]

metric=100*sum_Rn_P/sum_Rn
#print(metric)
MinCQM=min(metric[0])
if MinCQM<80:
    cResult='Failed'
else:
    cResult='Pass'
print('Causality check result: %s! (The CQM value is %5.3f%%)' %(cResult,MinCQM))

# Check the Reciprocity
# by Calculating the RQM
rs = np.zeros(shape=(frequencies,N,N))
RM = np.zeros(shape=(frequencies,1,1))

for m in range(frequencies):
    for i in range(N):
        for j in range(N):
           rs[m][i][j]=abs(m_d[m][i][j]-m_d[m][j][i])
           RM[m]=sum(sum(rs[m]))/(N*N)

for rr in RM:
    if rr<1e-6:
        rr=0
    else:
        RW=(rr-1e-6)/0.1

RQM=max(100/frequencies*(frequencies-sum(RW)),0)
if RQM<99:
    rResult='Failed'
else:
    rResult='Pass'
print('Reciprocity check result: %s! (The RQM value is %5.3f%%)' %(rResult,RQM))
print('\n----------------------------------------------------------\n')
# End