import numpy as np


class Simple_Linear_Regression:

    def __init__(self,input,target,epoch=100,learnrate=0.05):
        self.input = input
        self.target = target
        self.weight = np.array([np.zeros(input[0].size+1,dtype='float16')])
        self.epoch = epoch
        self.learnrate = learnrate

    def output(self,input):
        output = self.weight[:,1:].dot(input.T)+self.weight[:,0]
        self.output_=np.array([])
        self.output_=np.append(self.output_,output)
        return output

    def update(self,w,t,input):
        n = t.size
        o = self.output(input)
        delta_w = 2*self.learnrate/n*(t-o).dot(input)
        delta_w0 = 2*self.learnrate/n*(t-o).sum()
        w[:,1:] = w[:,1:] + delta_w
        w[:,0] = w[:,0] + delta_w0
        return w

    def train(self):
        for _ in range(self.epoch):
            self.weight = self.update(self.weight, self.target, self.input)
