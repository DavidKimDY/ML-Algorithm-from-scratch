import numpy as np


class Simple_Linear_Regression:

    def __init__(self,input,target,epoch=100,learnrate=0.05):
        self.input = input
        self.target = target
        self.weight = np.zeros(input[0].size+1)
        self.epoch = epoch
        self.learnrate = learnrate

    def cost_function(self):
        pass

    def output(self,input):
        output = input.dot(self.weight[1:])+self.weight[0]
        self.output_=[]
        self.output_.append(output)
        return output

    def update(self,w,t,input):
        n = t.size
        o = self.output(input)

        delta_w = 2*self.learnrate/n*(t-o).T.dot(input)
        delta_w = delta_w.sum()
        delta_w0 = 2*self.learnrate/n*(t-o)
        delta_w0 = delta_w0.sum()

        w[1:] = w[1:] + delta_w
        w[0] = w[0] + delta_w0

        return w

    def train(self):

        for _ in range(self.epoch):
            self.weight = self.update(self.weight, self.target, self.input)
            print(self.weight)
