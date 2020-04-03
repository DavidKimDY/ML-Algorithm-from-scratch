import numpy as np
class Adaline:
    def __init__(self, lr = 0.05 , lt = 10):
        self.lr = lr
        self.lt = lt

    def train(self, x, y):
        self.sse_ = []
        self.w_ = np.zeros(1+x.shape[1])
        exp = 1
        for i in range(self.lt):
            if i == exp:
                #print(self.w_)
                exp *= 3
            self.w_[1:] = self.w_[1:]+ self.delta_w(x,y)
        print('A_weight :\n', self.w_)

    def net_input(self,x):
        output =np.array([[]])
        for xi in x:
           output = np.append(output, np.dot(xi,self.w_.T[1:]))
        return output

    def predict(self,x):
        return np.where(self.net_input(x)>=0.0, 1 , -1)

    count=0
    exp = 1
    def delta_w(self,x,target):
        output = self.net_input(x)
        error = np.subtract(target, output)
        delta_sse = np.dot(error , x)
        self.w_[0] = error.sum()*self.lr
        self.output_ = output
        self.count +=1
        #if self.count == self.exp:
        self.sse_.append(error.sum())
        #    self.exp *= 3

        return delta_sse*self.lr
