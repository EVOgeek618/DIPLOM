import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import plotly
import plotly.graph_objs as go
import plotly.tools as tls
def VectorLen(Vector):
    return pow(Vector[0] ** 2 + Vector[1] ** 2 + Vector[2] ** 2, 0.5)
def Points2Vector(Pnt,i):
    Points = Pnt
    VP = i.split('-')
    Vector = np.array(Points[int(VP[1])]) - np.array(Points[int(VP[0])])
    VecLen = VectorLen(Vector)
    OrtVec = Vector / VecLen
    if VP[0]=='0':
        OrtVec *= 3
    return OrtVec
def OrtHand(Ort):
    HN = [0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
    Hand = [np.array([0, 0, 0])]
    for i in range(20):
        Hand.append(np.array(Hand[HN[i]]) + np.array(Ort[i]))
    Hand = np.array(Hand)
    return Hand
def Axis(Vector1, Vector2):
    return np.arccos(np.sum(Vector1 * Vector2) / (VectorLen(Vector1) * VectorLen(Vector2)))
def RightArea(Zhests, f, l):
    Vec1 = Zhests[f][5]
    Vec2 = Zhests[l][5]
    Cr = np.cross(Vec1, Vec2) / VectorLen(np.cross(Vec1, Vec2))
    th = Axis(Vec1, Vec2)
    c = np.cos(th)
    s = np.sin(th)
    [x, y, z] = Cr
    CRS = np.array([[c + (x ** 2) * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
                    [y * x * (1 - c) + z * s, c + (y ** 2) * (1 - c), y * z * (1 - c) - x * s],
                    [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + (z ** 2) * (1 - c)]])
    for i in range(1, 21):
        Zhests[f][i] = np.dot(CRS, Zhests[f][i])
    Cr = Zhests[l][5] / VectorLen(Zhests[l][5])
    Cr1 = Cr * VectorLen(Zhests[f][17]) * np.cos(Axis(Cr, Zhests[f][17]))
    Cr2 = Cr * VectorLen(Zhests[l][17]) * np.cos(Axis(Cr, Zhests[l][17]))
    th1 = Axis(Zhests[f][17] - Cr1, Zhests[l][17] - Cr2)
    c = np.cos(th1)
    s = np.sin(th1)
    [x, y, z] = Cr
    CRS = np.array([[c + (x ** 2) * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
                    [y * x * (1 - c) + z * s, c + (y ** 2) * (1 - c), y * z * (1 - c) - x * s],
                    [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + (z ** 2) * (1 - c)]])
    for i in range(1, 21):
        Zhests[f][i] = np.dot(CRS, Zhests[f][i])
    return th, th1
def Finger(Zhest,f,d=''):
    figs = []
    k=0
    FigName = ['Большой','Указательный','Средний','Безымянный','Мизинец']
    for (i,j) in [(1,5),(5,9),(9,13),(13,17),(17,21)]:
        fig = go.Scatter3d(x=([0]+list(Zhest[f][i:j, 0])),
                           y=([0]+list(Zhest[f][i:j, 1])),
                           z=([0]+list(Zhest[f][i:j, 2])),
                           name=FigName[k]+d)
        k += 1
        figs.append(fig)
    return figs
Data1 = pd.read_csv('HandsEtalon.csv',index_col=0)
'''Data = pd.read_csv('HandsTeach.csv',index_col=0)
Data1 = pd.read_csv('HandsTeach1.csv',index_col=0)
df1 = Data[:81]
df2 = Data1[81:]
Data1 = pd.concat([df1, df2], ignore_index=True)'''
Zhests = {}
for i in range(98):
    if Data1['isStatic'][i] == 1:
        Zhests[Data1['letter'][i]+str(i//33)] = np.array(eval(Data1['hands'][i]))
P2V = ['0-1', '1-2', '2-3', ' 3-4', '0-5', '5-6', '6-7', '7-8', '0-9', '9-10', '10-11',
           '11-12', '0-13', '13-14', '14-15', '15-16', '0-17', '17-18', '18-19', '19-20'] #Индексы точек отдельных векторов
frame = cv2.imread('O.png')
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
points=[]
for i in results.multi_hand_landmarks[0].landmark:
        point = [i.x,i.y,i.z]
        points.append(point)
Vectors = []
for j in P2V:
    OrtVec = Points2Vector(points,j)
    Vectors.append(OrtVec)
Et = {}
Ort = np.array(OrtHand(Vectors))
Zhests['Now'] = Ort
for i in ['В0']:
    if i == 'В1':
        continue
    th,th1 = RightArea(Zhests,i,'В1')
fig1 = Finger(Zhests,'В0')
fig2 = Finger(Zhests,'В1','1')
figs = []
LET = list(set([i[:-1] for i in Zhests.keys()]))
LET.sort()
LET.remove('No')
print(LET)
F = Zhests['А0'][:, 0].tolist()
'''for i in ['Я', 'Б']:
    try:
        fig = go.Scatter3d(x=(Zhests[i+'0'][:, 0].tolist()+Zhests[i+'1'][:, 0].tolist()+Zhests[i+'2'][:, 0].tolist()),
                                   y=(Zhests[i+'0'][:, 1].tolist()+Zhests[i+'1'][:, 1].tolist()+Zhests[i+'2'][:, 1].tolist()),
                                   z=(Zhests[i+'0'][:, 2].tolist()+Zhests[i+'1'][:, 2].tolist()+Zhests[i+'2'][:, 2].tolist()),
                                   mode='markers',
                                   name=i)
    except KeyError:
        fig = go.Scatter3d(x=(Zhests[i+'0'][:, 0].tolist()+Zhests[i+'1'][:, 0].tolist()),
                           y=(Zhests[i+'0'][:, 1].tolist()+Zhests[i+'1'][:, 1].tolist()),
                           z=(Zhests[i+'0'][:, 2].tolist()+Zhests[i+'1'][:, 2].tolist()),
                           mode='markers',
                           name=i)
    figs.append(fig)
t = 12
for i in LET:
    try:
        fig = go.Scatter3d(x=([Zhests[i+'0'][t, 0]]+[Zhests[i+'1'][t, 0]]+[Zhests[i+'2'][t, 0]]),
                                   y=([Zhests[i+'0'][t, 1]]+[Zhests[i+'1'][t, 1]]+[Zhests[i+'2'][t, 1]]),
                                   z=([Zhests[i+'0'][t, 2]]+[Zhests[i+'1'][t, 2]]+[Zhests[i+'2'][t, 2]]),
                                   mode='markers',
                                   name=i)
    except KeyError:
        fig = go.Scatter3d(x=([Zhests[i+'0'][t, 0]]+[Zhests[i+'1'][t, 0]]),
                           y=([Zhests[i+'0'][t, 1]]+[Zhests[i+'1'][t, 1]]),
                           z=([Zhests[i+'0'][t, 2]]+[Zhests[i+'1'][t, 2]]),
                           mode='markers',
                           name=i)
    figs.append(fig)
fig = go.Scatter3d(x=([Ort[t,0]]),
                           y=([Ort[t,1]]),
                           z=([Ort[t,2]]),
                           mode='markers',
                           name='Жест')
theta = np.linspace(0,2*3.14159,100)
phi = np.linspace(0,3.14159,100)
x = np.outer(2*np.cos(theta),2*np.sin(phi))+np.ones((100,100))*Ort[t,0]
y = np.outer(2*np.sin(theta),2*np.sin(phi))+np.ones((100,100))*Ort[t,1]
z = np.outer(2*np.ones(100),2*np.cos(phi))+np.ones((100,100))*Ort[t,2] # note this is 2d now
trace1 = go.Surface(x=x,y=y,z=z,opacity=0.4,showlegend=True)
figs.append(fig)
figs.append(trace1)'''
mylayout = go.Layout(scene=dict(xaxis=dict(title="X"),
                                    yaxis=dict(title="Y"),
                                    zaxis=dict(title="Z")))
i = 'ZhestsOrt'
plotly.offline.plot({"data": fig1+fig2,
                "layout": mylayout},
                auto_open=True,
                filename='Dip'+str(i)+'.html')