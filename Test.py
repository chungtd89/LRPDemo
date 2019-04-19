'''
Created on Apr 13, 2019

@author: chungtd6
'''
import numpy, copy, modules, utils

class NextLinear(modules.Linear):
    def relprop(self,R):
        V = numpy.maximum(0,self.W)
        Z = numpy.dot(self.X,V)+1e-9; S = R/Z
        C = numpy.dot(S,V.T);         R = self.X*C
        return R
    
class FirstLinear(modules.Linear):
    def relprop(self,R):
        W,V,U = self.W,numpy.maximum(0,self.W),numpy.minimum(0,self.W)
        X,L,H = self.X,self.X*0+utils.lowest,self.X*0+utils.highest

        Z = numpy.dot(X,W)-numpy.dot(L,V)-numpy.dot(H,U)+1e-9; S = R/Z
        R = X*numpy.dot(S,W.T)-L*numpy.dot(S,V.T)-H*numpy.dot(S,U.T)
        return R
    
nn = modules.Network([
    modules.Linear('mlp/l1'),modules.ReLU(),
    modules.Linear('mlp/l2'),modules.ReLU(),
    modules.Linear('mlp/l3'),
    ])

X,T = utils.getMNISTsample(N=12,path='mnist/',seed=1234)
utils.visualize(X,utils.graymap,'data.png')

Y = nn.forward(X)
S = nn.gradprop(T)**2
utils.visualize(S,utils.heatmap,'mlp-sensitivity.png')

nn = modules.Network([
    FirstLinear('mlp/l1'),modules.ReLU(),
    NextLinear('mlp/l2'),modules.ReLU(),
    NextLinear('mlp/l3'),modules.ReLU(),
])

Y = nn.forward(X)
D = nn.relprop(Y*T)
utils.visualize(D,utils.heatmap,'mlp-deeptaylor.png')



#CNN
cnn = modules.Network([
    modules.Convolution('cnn/c1-5x5x1x10'),modules.ReLU(),modules.Pooling(),
    modules.Convolution('cnn/c2-5x5x10x25'),modules.ReLU(),modules.Pooling(),
    modules.Convolution('cnn/c3-4x4x25x100'),modules.ReLU(),modules.Pooling(),
    modules.Convolution('cnn/c4-1x1x100x10'),
])

class NextConvolution(modules.Convolution):
    def relprop(self,R):
        pself = copy.deepcopy(self); pself.B *= 0; pself.W = numpy.maximum(0,pself.W)

        Z = pself.forward(self.X)+1e-9; S = R/Z
        C = pself.gradprop(S);          R = self.X*C
        return R
    
class FirstConvolution(modules.Convolution):
    def relprop(self,R):
        iself = copy.deepcopy(self); iself.B *= 0
        nself = copy.deepcopy(self); nself.B *= 0; nself.W = numpy.minimum(0,nself.W)
        pself = copy.deepcopy(self); pself.B *= 0; pself.W = numpy.maximum(0,pself.W)
        X,L,H = self.X,self.X*0+utils.lowest,self.X*0+utils.highest

        Z = iself.forward(X)-pself.forward(L)-nself.forward(H)+1e-9; S = R/Z
        R = X*iself.gradprop(S)-L*pself.gradprop(S)-H*nself.gradprop(S)
        return R
    
class Pooling(modules.Pooling):
    def relprop(self,R):
        Z = (self.forward(self.X)+1e-9); S = R / Z
        C = self.gradprop(S);            R = self.X*C
        return R
    
X,T = utils.getMNISTsample(N=12,path='mnist/',seed=1234)

padding = ((0,0),(2,2),(2,2),(0,0))
X = numpy.pad(X.reshape([12,28,28,1]),padding,'constant',constant_values=(utils.lowest,))
T = T.reshape([12,1,1,10])

Y = cnn.forward(X)
S = cnn.gradprop(T)**2
utils.visualize(S[:,2:-2,2:-2],utils.heatmap,'cnn-sensitivity.png')

cnn = modules.Network([
    FirstConvolution('cnn/c1-5x5x1x10'),modules.ReLU(),Pooling(),
    NextConvolution('cnn/c2-5x5x10x25'),modules.ReLU(),Pooling(),
    NextConvolution('cnn/c3-4x4x25x100'),modules.ReLU(),Pooling(),
    NextConvolution('cnn/c4-1x1x100x10'),modules.ReLU(),
])

Y = cnn.forward(X)
D = cnn.relprop(Y*T)
utils.visualize(D[:,2:-2,2:-2],utils.heatmap,'cnn-deeptaylor.png')

#AlphaBeta
class NextConvolutionAlphaBeta(modules.Convolution,object):

    def __init__(self,name,alpha):
        super(self.__class__, self).__init__(name)
        self.alpha = alpha
        self.beta  = alpha-1
        
    def relprop(self,R):
        pself = copy.deepcopy(self); pself.B *= 0; pself.W = numpy.maximum( 1e-9,pself.W)
        nself = copy.deepcopy(self); nself.B *= 0; nself.W = numpy.minimum(-1e-9,nself.W)

        X = self.X+1e-9
        ZA = pself.forward(X); SA =  self.alpha*R/ZA
        ZB = nself.forward(X); SB = -self.beta *R/ZB
        R = X*(pself.gradprop(SA)+nself.gradprop(SB))
        return R

cnn = modules.Network([
    FirstConvolution('cnn/c1-5x5x1x10'),modules.ReLU(),Pooling(),
    NextConvolutionAlphaBeta('cnn/c2-5x5x10x25',2.0),modules.ReLU(),Pooling(),
    NextConvolutionAlphaBeta('cnn/c3-4x4x25x100',2.0),modules.ReLU(),Pooling(),
    NextConvolutionAlphaBeta('cnn/c4-1x1x100x10',2.0),modules.ReLU(),
])

Y = cnn.forward(X)
D = cnn.relprop(Y*T)
utils.visualize(D[:,2:-2,2:-2],utils.heatmap,'cnn-alphabeta.png')



