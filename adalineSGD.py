import numpy as np
class AdalineSGD:
    def __init__(self, lr = 0.05 , lt = 10):
        self.lr = lr
        self.lt = lt

    def train(self, x, y):
        self.sse_ = []
        self.w_ = np.zeros(1+x.shape[1])
        exp=1
        for i in range(self.lt):
            for xi,target in zip(x,y.T):
                self.w_[1:] = self.w_[1:]+ self.delta_w(xi,target)

        print('AS_weight :\n', self.w_)

    def net_input(self,xi):
        output = np.dot(xi,self.w_.T[1:])
        return output

    def predict(self,x):
        return np.where(self.net_input(x)>=0.0, 1 , -1)

    count=0
    exp = 1
    def delta_w(self,xi,target):
        output = self.net_input(xi)
        error = target - output

        self.count +=1
        if self.count == self.exp:
            self.sse_.append(error)
            self.exp *= 3

        delta_sse = error * xi
        self.w_[0] = error*self.lr
        return delta_sse*self.lr
