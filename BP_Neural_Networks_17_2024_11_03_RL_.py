# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 22:15:54 2023
@author: Dongyuan Ge
"""
import numpy as np
import math
import random
import string
import matplotlib as mpl
import matplotlib.pyplot as plt



# 生成区间[a,b]内的随机数
def random_number(a, b):
    return (b - a) * random.random() + a


# 生成一个矩阵，大小为m*n,并且设置默认零矩阵
def makematrix(m, n, fill=0.0):
    a = []
    for i in range(m):
        a.append([fill] * n)
    return a


# 函数sigmoid(),这里采用tanh，因为看起来要比标准的sigmoid函数好看
def sigmoid(x):
    return 1*(math.tanh(x))


# 函数sigmoid的派生函数
def derived_sigmoid(x):
    return 1*(1.0 - x ** 2)


# 构造三层BP网络架构
class BPNN:
    def __init__(self, num_in, num_hidden, num_out):
        # 输入层，隐藏层，输出层的节点数
        self.num_in = num_in   # 增加一个偏置结点
        self.num_hidden = num_hidden   # 增加一个偏置结点
        self.num_out = num_out

        # 激活神经网络的所有节点（向量）
        self.active_in = [1.0] * self.num_in
        self.active_hidden = [1.0] * self.num_hidden
        self.active_out = [1.0] * self.num_out   #可以用2

        # 创建权重矩阵
        self.wight_in = makematrix(self.num_in, self.num_hidden)
        self.wight_out = makematrix(self.num_hidden, self.num_out)

        #对权值矩阵赋初值
        for i in range(self.num_in):
            for j in range(self.num_hidden):
                self.wight_in[i][j] = random_number(-0.9, 0.9)   #原来的是random_number(-0.2, 0.2)
        for j in range(self.num_hidden):
            for k in range(self.num_out):
                self.wight_out[j][k] = random_number(-0.9, 0.9)

        self.wight_in=[

            [5.440239303807346, -7.802442300880394, 3.4006028921087763, 0.035213945686107175, 1.224034468707315,  -0.3082911644841659,  4.914494182297433, -7.414199344570143, 3.0999550056010245, -0.05260641890896131, 0.5837915094262986, 0.3719391110041306],
            [0.12756678562558468, -5.012895033040633, -0.8513042426412959, 1.0173955946889524, -3.905490842733331, -0.8265196267362495, 0.11298067504309889, -4.383979924412649, -0.8798795189532688, 0.5391518851734661, -3.0024128916996573, -1.0030683795037518],
            [-0.08900442156327965, 4.289381170264661, -26.02578208293187, 15.616363623957339, 9.913510635027906,   6.333925420269499,  -0.34784472541339023, 4.694637776012748, -25.58076601190935, 15.646544970204605, 9.062552490444087, 6.510054291245681],
            [-0.33298558167661846, 0.5570382970007148, -1.891984526805138, -0.48282010298878936, -0.9438004313569472,  -0.26154798984901684,  -0.3312280523557268, 0.5586027320405468, -1.890066835244081, -0.4788526826171701, -0.9491610802445143, -0.2584008640178758]
             # 2024.10.05       前两次运行得到的权值的组合，前面的4*6是左摄像机的权值，后面的4*6是右摄像机的权值
            ]
        self.wight_out = [
            # 以下为左摄像机的
            [1.473425516227368, 1.7760682694755539, 0, 0],
            [0.58108139410643, 0.6079339781191161, 0,  0],
            [2.2886828886051607, -0.009777319246751508, 0  ,0],
            [3.966756592249999, 0.5940259778771153,  0,  0],
            [0.3533953579273661, -4.048554884901139,  0,  0],
            [-3.5309519775935367, 4.495948409559406,  0,  0],
#以下为右摄像机的
            [0, 0, 1.5316266604846562, 0.17873267570618354],
            [0, 0, 0.31434472800150054, 0.08281598724383565],
            [0, 0, 1.573530858287362, 0.14450789704943232],
            [0, 0, 2.8015943622603654, 0.7479922369749935],
            [0, 0, 0.8097699174581005, -3.643706936135235],
            [0, 0, -3.313947888587198, 3.7749850769537834]
            # 2024.10.05      前两次运行得到的权值的组合
        ]
        # 最后建立动量因子（矩阵）
        self.ci = makematrix(self.num_in, self.num_hidden)
        self.co = makematrix(self.num_hidden, self.num_out)

        # 信号正向传播

    def update(self, inputs):
        if len(inputs) != self.num_in:
            raise ValueError('与输入层节点数不符')

        # 数据输入输入层
        for i in range(self.num_in ):
                 self.active_in[i] = inputs[i]  # active_in[]是输入数据的矩阵

        # 数据在隐藏层的处理
        for j in range(self.num_hidden ):
            sum = 0.0
            for i in range(self.num_in):
                sum = sum + self.active_in[i] * self.wight_in[i][j]
            self.active_hidden[j] = sigmoid(sum)  # active_hidden[]是处理完输入数据之后存储，作为输出层的输入数据

        # 数据在输出层的处理
        for k in range(self.num_out):
            sum = 0.0
            for j in range(self.num_hidden):
                sum = sum + self.active_hidden[j] * self.wight_out[j][k]
            self.active_out[k] = sigmoid(sum)  # 与上同理

        return self.active_out[:]

    # 误差反向传播
    def errorbackpropagate(self, targets, lr, m):  # lr是学习率， m是动量因子
        if len(targets) != self.num_out:
            raise ValueError('与输出层节点数不符！')

        # 首先计算输出层的误差
        out_deltas = [0.0] * self.num_out
        for k in range(self.num_out):
            error = targets[k] - self.active_out[k]
            out_deltas[k] = derived_sigmoid(self.active_out[k]) * error

        # 然后计算隐藏层误差
        hidden_deltas = [0.0] * self.num_hidden
        for j in range(self.num_hidden):
            error = 0.0
            for k in range(self.num_out):
                error = error + out_deltas[k] * self.wight_out[j][k]
            hidden_deltas[j] = derived_sigmoid(self.active_hidden[j]) * error

        # 然后更新输入的信息
        for i in range(self.num_in-1):
            change=[0, 0, 0]
            for j in range(self.num_hidden):
                change[i] =change[i]+ hidden_deltas[j] * self.wight_in[i][j]

            self.active_in[i] = self.active_in[i] + lr * change[i]
                #self.ci[i][j] = change

        # 计算总误差
        error = 0.0
        for i in range(len(targets)):
            error = error + 0.5 * (targets[i] - self.active_out[i]) ** 2
        return error

    # 测试
    def test(self, patterns):
        for i in patterns:
            print(i[0], '->', self.update(i[0]))

    # 权重
    def weights(self):
        print("输入层权重")
        for i in range(self.num_in):
            print(self.wight_in[i])
        print("输出层权重")
        for i in range(self.num_hidden):
            print(self.wight_out[i])


    def train(self, pattern, itera=300001, lr=0.00200, m=0.00):

        #for j in pattern:
        for k, j in enumerate( pattern):
            for i in range(itera):  # 9500000
                error = 0.0
                if i == 0:
                   inputs = j[0]
                #   print('initial1=',inputs)
                   # inputs=[0.9698416053274535, 0.9492286353372284, 0.7490594680050898, 0.0001]
                   # print('initial2=', inputs)
                   targets = j[1]
                   self.update(inputs)
                   error = error + self.errorbackpropagate(targets, lr, m)
                else:
            # for j in pattern:
                   inputs =[self.active_in[0],self.active_in[1],self.active_in[2],0.0001]
                   # print('gdy_1030=', inputs)
                   targets = j[1]
                   self.update(inputs)
                   error = error + self.errorbackpropagate(targets, lr, m)
                if i % 300000 == 0:
                   # print('误差gdy %-.25f'  error)
                   # print('gdy_inputs=', inputs)
                   # print(  error)
                   # print( inputs)
                   print(k+1,error, inputs)
                 #  print(inputs)

# 实例
def demo():
    patt  =  [
    #L1    [[random.random(), random.random(), random.random(), 0.0001], [0.172734197903692, 0.341551643114009, 0.109959927, 0.269760103]],
    #    [[random.random(), random.random(), random.random(), 0.0001], [0.136198885909994, 0.343503030766505, 0.079159675, 0.270884905]],
    #    [[random.random(), random.random(), random.random(), 0.0001],[0.044994830446407, 0.233664846916856, 0.013172624, 0.184271746]],     #第一行的数据
    #   [[random.random(), random.random(), random.random(), 0.0001], [0.065738417769928,	0.231975132530718,	0.030108340,	0.183751941]],  # 第二行的数据
    #    [[random.random(), random.random(), random.random(), 0.0001], [0.086497335717812,	0.230217059610868,	0.047006554,	0.183233288 ]],  # 第三行的数据
    #     [[random.random(), random.random(), random.random(), 0.0001],[0.159523787436676,	0.351662100885110,	0.100722796,	0.282555302]],  # 第48行的数据
    #
    #     [[random.random(), random.random(), random.random(), 0.0001], [0.074851025000627,	0.292102734244481,	0.031472279,	0.230067125]],  # 第72行的数据

        [[random.random(), random.random(), random.random(), 0.0001],
         [0.044994830, 0.233664847, 0.012865384, 0.184351886]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.065738418, 0.231975133, 0.029783653, 0.183793683]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.086497336, 0.230217060, 0.046694583, 0.183244996]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.107232290, 0.228473626, 0.063586770, 0.182706531]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.127936360, 0.226792792, 0.080448855, 0.182178972]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.148625926, 0.225022892, 0.097269550, 0.181662972]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.169297668, 0.223292219, 0.114037655, 0.181159152]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.046435456, 0.254506972, 0.013357191, 0.201157488]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.067194705, 0.252748248, 0.030280384, 0.200578317]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.087999144, 0.251010981, 0.047199285, 0.200009411]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.108820126, 0.249237463, 0.064102366, 0.199451449]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.129638314, 0.247497463, 0.080978118, 0.198905084]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.150336769, 0.245740189, 0.097815086, 0.198370934]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.171130991, 0.244060026, 0.114601884, 0.197849584]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.047890625, 0.275473811, 0.013836418, 0.218009955]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.068684907, 0.273722250, 0.030766770, 0.217411537]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.089585384, 0.271952168, 0.047695821, 0.216824095]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.110440872, 0.270189420, 0.064611905, 0.216248281]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.131238078, 0.268467101, 0.081503355, 0.215684711]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.152158619, 0.266637040, 0.098358538, 0.215133971]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.172960802, 0.264925901, 0.115165873, 0.214596607]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.049310725, 0.296621799, 0.014304877, 0.234894407]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.070185066, 0.294827236, 0.031244585, 0.234278490]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.091139013, 0.293059401, 0.048185921, 0.233674223]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.112082006, 0.291295329, 0.065117068, 0.233082222]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.132949757, 0.289538993, 0.082026193, 0.232503074]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.153894378, 0.287766053, 0.098901473, 0.231937324]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.174885634, 0.286023472, 0.115731126, 0.231385482]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.050743626, 0.317734276, 0.014764384, 0.251795677]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.071686990, 0.315954076, 0.031715605, 0.251164033]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.092644772, 0.314226265, 0.048671312, 0.250544668]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.113702678, 0.312469527, 0.065619531, 0.249938169]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.134742833, 0.310678305, 0.082548250, 0.249345083]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.155735502, 0.308934875, 0.099445449, 0.248765921]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.176833649, 0.307268695, 0.116299136, 0.248201150]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.052201285, 0.339042015, 0.015216755, 0.268698368]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.073245566, 0.337242591, 0.032181599, 0.268052782]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.094253403, 0.335394282, 0.049153715, 0.267420064]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.115358959, 0.333673784, 0.066120960, 0.266800764]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.136499532, 0.331942794, 0.083071133, 0.266195394]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.157628077, 0.330223762, 0.099992010, 0.265604425]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.178775923, 0.328539015, 0.116871380, 0.265028285]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.053733328, 0.360192781, 0.015663801, 0.285586908]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.074795437, 0.358442074, 0.032644333, 0.284929174]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.095908997, 0.356731321, 0.049634843, 0.284284852]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.117095238, 0.354970704, 0.066623011, 0.283654455]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.138340936, 0.353307863, 0.083596437, 0.283038458]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.159523787, 0.351662101, 0.100542686, 0.282437292]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.180796766, 0.349921699, 0.117449321, 0.281851345]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.052459242, 0.234999936, 0.014216980, 0.182396742]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.071642037, 0.233752730, 0.030357940, 0.181875845]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.090866000, 0.232463273, 0.046476225, 0.181362137]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.110042189, 0.231179955, 0.062561863, 0.180856356]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.129164368, 0.229866645, 0.078604996, 0.180359218]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.148347532, 0.228548863, 0.094595890, 0.179871419]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.167424477, 0.227236101, 0.110524954, 0.179393626]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.053481223, 0.254329766, 0.014694813, 0.198420007]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.072695324, 0.253014084, 0.030839759, 0.197875130]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.091985878, 0.251728187, 0.046965223, 0.197338255]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.111224940, 0.250355804, 0.063061125, 0.196810094]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.130463067, 0.249013060, 0.079117475, 0.196291336]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.149601463, 0.247709396, 0.095124388, 0.195782645]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.168738067, 0.246361787, 0.111072101, 0.195284654]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.054474493, 0.273840487, 0.015155518, 0.214512450]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.073770281, 0.272515482, 0.031306754, 0.213945210]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.093126749, 0.271163283, 0.047441655, 0.213386750]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.112405113, 0.269785233, 0.063550021, 0.212837756]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.131706682, 0.268453880, 0.079621719, 0.212298886]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.150917418, 0.267102945, 0.095646701, 0.211770771]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.170273086, 0.265714626, 0.111615025, 0.211254011]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.055469769, 0.293524623, 0.015600846, 0.230660818]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.074851025, 0.292102734, 0.031760642, 0.230072868]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.094264899, 0.290770764, 0.047907199, 0.229494439]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.113655647, 0.289368320, 0.064030184, 0.228926189]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.133029290, 0.288027189, 0.080119309, 0.228368745]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.152344604, 0.286641996, 0.096164357, 0.227822705]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.171712276, 0.285295834, 0.112155195, 0.227288629]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.056503166, 0.313326872, 0.016032550, 0.246851513]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.075966045, 0.311893798, 0.032203142, 0.246244533]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.095396196, 0.310526395, 0.048363530, 0.245647780]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.114887867, 0.309139902, 0.064503241, 0.245061878]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.134338956, 0.307746793, 0.080611823, 0.244487424]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.153786814, 0.306406773, 0.096678875, 0.243924978]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.173245102, 0.305044765, 0.112694068, 0.243375063]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.057517625, 0.333117859, 0.016452388, 0.263070627]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.077041491, 0.331721201, 0.032635969, 0.262446326]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.096547514, 0.330296612, 0.048812322, 0.261832914]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.116110686, 0.328986707, 0.064970815, 0.261230986]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.135666181, 0.327571798, 0.081100826, 0.260641103]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.155148835, 0.326277342, 0.097191764, 0.260063789]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.174679537, 0.324915295, 0.113233091, 0.259499528]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.058598142, 0.353179443, 0.016862117, 0.279303993]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.078197469, 0.351662510, 0.033060842, 0.278664095]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.097767267, 0.350255230, 0.049255241, 0.278035709]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.117381940, 0.348929052, 0.065434523, 0.277419396]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.137021468, 0.347559513, 0.081587880, 0.276815680]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.156629991, 0.346203607, 0.097704521, 0.276225049]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.176266917, 0.344930188, 0.113773697, 0.275647947]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.059365558, 0.238133298, 0.014659874, 0.180928082]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.077260251, 0.237347595, 0.030173349, 0.180463656]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.095145242, 0.236602188, 0.045649870, 0.180004089]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.113021437, 0.235808765, 0.061080502, 0.179550137]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.130893521, 0.235001366, 0.076456482, 0.179102544]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.148690190, 0.234139275, 0.091769221, 0.178662037]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.166449839, 0.233233337, 0.107010323, 0.178229322]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.059959096, 0.255870121, 0.015102776, 0.196126916]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.077884987, 0.255067670, 0.030618850, 0.195635780]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.095863375, 0.254211019, 0.046101305, 0.195150372]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.113783504, 0.253355095, 0.061541115, 0.194671428]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.131738497, 0.252533427, 0.076929403, 0.194199667]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.149553543, 0.251620797, 0.092257446, 0.193735789]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.167394838, 0.250709866, 0.107516691, 0.193280468]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.060490112, 0.273843546, 0.015524530, 0.211414504]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.078565165, 0.272982593, 0.031045565, 0.210898151]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.096572660, 0.272066592, 0.046536279, 0.210388367]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.114565017, 0.271191903, 0.061987544, 0.209885863]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.132537125, 0.270309278, 0.077390358, 0.209391332]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.150494335, 0.269374119, 0.092735852, 0.208905445]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.168416752, 0.268439555, 0.108015310, 0.208428842]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.061176329, 0.292091584, 0.015926818, 0.226779249]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.079276130, 0.291091558, 0.031455147, 0.226239214]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.097331166, 0.290185749, 0.046956412, 0.225706555]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.115410389, 0.289231787, 0.062421370, 0.225181959]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.133388573, 0.288318037, 0.077840882, 0.224666090]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.151417252, 0.287337227, 0.093205926, 0.224159586]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.169395655, 0.286401793, 0.108507609, 0.223663056]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.061715802, 0.310392522, 0.016311329, 0.242209165]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.079902788, 0.309424530, 0.031849254, 0.241647018]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.098023475, 0.308478434, 0.047363323, 0.241093021]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.116115329, 0.307531686, 0.062844169, 0.240547832]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.134251939, 0.306565738, 0.078282506, 0.240012086]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.152378483, 0.305573976, 0.093669146, 0.239486389]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.170516754, 0.304605187, 0.108995011, 0.238971313]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.062301212, 0.328882913, 0.016679755, 0.257691910]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.080556458, 0.327926991, 0.032229543, 0.257109254]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.098752037, 0.326928856, 0.047758632, 0.256535483]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.116984280, 0.325924705, 0.063257517, 0.255971229]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.135149725, 0.324933793, 0.078716755, 0.255417094]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.153448408, 0.323943542, 0.094126981, 0.254873649]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.171590141, 0.322971325, 0.109478927, 0.254341429]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.062892549, 0.347529296, 0.017033794, 0.273214818]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.081190412, 0.346499638, 0.032597677, 0.272613281]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.099494159, 0.345472341, 0.048143958, 0.272021328]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.117873578, 0.344462071, 0.063662984, 0.271439558]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.136198886, 0.343503031, 0.079145146, 0.270868543]],
        [[random.random(), random.random(), random.random(), 0.0001],
         [0.154447744, 0.342554010, 0.094580893, 0.270308816]],
        [[random.random(), random.random(), random.random(), 0.0001],    [0.172734198, 0.341551643, 0.109960757, 0.269760876]],









    ]

    # 创建神经网络，4个输入节点，12个隐藏层节点，4个输出层节点
    n = BPNN(4, 12, 4)
    # 训练神经网络
    n.train(patt)
    # 测试神经网络
    # n.test(patt)
    # 查阅权重值
    n.weights()


if __name__ == '__main__':
    demo()