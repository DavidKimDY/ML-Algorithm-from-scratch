import numpy as np
class Perceptron:
    def __init__(self, lr = 0.05 , lt = 10):
        self.lr = lr
        self.lt = lt

    def train(self, x, target):
        self.w_ = np.zeros(1+x.shape[1])
        for _ in range(self.lt):
            for i in range(x.T[0].size):
                output = self.predict(x[i])
                error = target[:,i] - output
                self.w_[0] += self.lr * error
                self.w_[1:] += self.lr * error * x[i]
        w = np.copy(self.w_).astype('float16')
        print(f'P_weight :\n{w}' )

    def net_input(self,x):
        return np.dot(x,self.w_.T[1:])

    def predict(self,x):
        return np.where(self.net_input(x)>=0.0, 1 , -1)
