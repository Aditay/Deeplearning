# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import cPickle, gzip
import theano.tensor as T
import numpy as np
import theano

#%%
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()
#%%
train_set_x = train_set[0]
train_set_y = train_set[1]
test_set_x = test_set[0]
test_set_y = test_set[1]
valid_set_x = valid_set[0]
valid_set_y = valid_set[1]
#%%


#%%

def shared_dataset(data_xy):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX))
    shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX))
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets us get around this issue
    return shared_x, T.cast(shared_y, 'int32')

test_set_x, test_set_y = shared_dataset(test_set)
valid_set_x, valid_set_y = shared_dataset(valid_set)
train_set_x, train_set_y = shared_dataset(train_set)
#%%

def y2indicator(y):
    N = len(y)
    ind = np.zeros((N,10))
    for i in xrange(N):
        ind[i,y[i]] = 1
    return ind

def relu(x):
    return x*(x > 0)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(a):
    expA = np.exp(a)
    return expA/np.sum(expA,axis=1, keepdims=True)


#%%

train_set_ind = y2indicator(train_set_y)
test_set_ind = y2indicator(test_set_y)
val_set_ind = y2indicator(valid_set_y)

#%%

#W1 = T.matrix('W1')
#scaling1 = T.vector('S1')
#W2 = T.matrix('W2')
#scaling2 = T.vector('S2')
#W3 = T.matrix('W3')
#scaling3 = T.vector('S3')
#W4 = T.matrix('W4')
#scaling4 = T.vector('S4')
#W5 = T.matrix('W5')
#scaling5 = T.vector('S5')
#W6 = T.matrix('W6')
#scaling6 = T.vector('S6')
#outputMatrix = T.matrix('outputMatrix')

#def multiplyMatrices(inputMat,weight,scale):
#    product = 2*scale*T.dot(weight,inputMat)    
    
#    matrix_multiply = theano.function(inputs=[inputMat,weight,scale], output=[product])
#    product = matrix_multiply(inputMat,weight,scale)    
#    return product
def addLayer(inputMat,D,M):
    weight = theano.shared(np.random.randn(D,M))
#    if addScaling:
#        scale = theano.shared(np.random.randn(D),'scale')
#    else:
#        scale = theano.shared(np.random.randn(D),'scale')
#    actualScale = sigmoid(scale)
#    getScale = theano.function(inputs = [scale], outputs = [actualScale])
#    aScale = getScale(scale)   
#    hiddenLayer = multiplyMatrices(inputMatrix, weight, aScale)
    h = T.dot(weight, inputMat)
    bias = theano.shared(np.random.randn(D))
    hiddenLayer = relu(h+bias)    
#    return [hiddenLayer,weight,scale]
    return [hiddenLayer, weight,bias]

#%%
inputMatrix = T.matrix('inputVector')
targetMatrix = T.matrix('targetMatrix')

M = 784

D = 500

N = 10
[hidden1, W1, b1] = addLayer(inputMatrix,D,M)
[hidden2, W2, b2] = addLayer(hidden1,D,D)
[hidden3, W3, b3] = addLayer(hidden2,D,D)
[hidden4, W4, b4] = addLayer(hidden3,D,D)
[hidden5, W5, b5] = addLayer(hidden4,D,D)
[hidden6, W6, b6] = addLayer(hidden5,N,D)
#print hidden1
#print W1
#print b1
outputProbabilities = T.nnet.softmax(hidden6)
#%%
cost = -(targetMatrix*T.log(hidden6)).sum()
prediction = T.argmax(hidden6,axis=1)

lr = 0.02
update_W1 = W1 - lr*T.grad(cost,W1)
update_W2 = W2 - lr*T.grad(cost,W2)
update_W3 = W3 - lr*T.grad(cost,W3)
update_W4 = W4 - lr*T.grad(cost,W4)
update_W5 = W5 - lr*T.grad(cost,W5)
update_W6 = W6 - lr*T.grad(cost,W6)

update_b1 = b1 - lr*T.grad(cost,b1)
update_b2 = b2 - lr*T.grad(cost,b2)
update_b3 = b3 - lr*T.grad(cost,b3)
update_b4 = b4 - lr*T.grad(cost,b4)
update_b5 = b5 - lr*T.grad(cost,b5)
update_b6 = b6 - lr*T.grad(cost,b6)

#%%
train = theano.function(inputs = [inputMatrix,targetMatrix],
                        updates = [(W1,update_W1),(W2,update_W2),(W3,update_W3),(W4,update_W4),(W5,update_W5),(W6,update_W6), (b1,update_b1),(b2,update_b2),(b3,update_b3),(b4,update_b4),(b5,update_b5),(b6,update_b6)] ,
                        )

#%%

from theano import shared
state = shared(0)
inc = T.iscalar('inc')
accumulator = function([inc], state, updates=[(state, state+inc)])

#%%

 x = theano.tensor.dscalar()
 f = theano.function([x],
                     2*x
                     )
 f(4)
 print f(4)
#%%
get_prediction = theano.function( inputs = [inputMatrix,targetMatrix],outputs = [cost,outputProbabilities])


#%%
max_iter = 10000;
n_batch = 256
batch_sz = 256

## This is batch Stochastic Gradient Descent
for i in xrange(max_iter):
    for j in xrange(n_batch):
        Xbatch = Xtrain[j*batch_sz:(j*batch_sz+batch_sz),]
        Ybatch = Ytrain[j*batch_sz:(j*batch_sz+batch_sz),]
        
        train(Xbatch,Ybatch)
        if j% 10 == 0:
            cost_val, pred_val = get_prediction(inputMatrix,targetMatrix)
            print cost_val
            print pred_val
    
    
def addScale(weight,inputMatrix):
        scale = sigmoid(theano.shared(np.random.randn(D),'scale'))
        
        scaledHidden = 2*scale*T.dot(weight,inputMatrix)
        
        return [scale, scaledHidden]
        
[S1,SH1] = addScale(W1,inputMatrix)
[S2, SH2] = addScale(W2,SH1)
[S3, SH3] = addScale(W3,SH2)
[S4, SH4] = addScale(W4,SH3)
[S5, SH5] = addScale(W5,SH4)
[S6, SH6] = addScale(W6,SH5)

outputProb = T.nnet.softmax(SH6)
targetMatTest = T.matrix('test')
inputMatrixTest = T.matrix('inputTest')
costTest = -(targetMatTest*T.log(SH6,axis=1))

lr2 = 0.02
update_S1 = S1 - lr2*T.grad(cost,S1)
update_S2 = S2 - lr2*T.grad(cost,S2)
update_S3 = S3 - lr2*T.grad(cost,S3)
update_S4 = S4 - lr2*T.grad(cost,S4)
update_S5 = S5 - lr2*T.grad(cost,S5)
update_S6 = S6 - lr2*T.grad(cost,S6)

testFine = theano.function(
    inputs = [inputMatrixTest,targetMatTest],
    updates = [(S1,update_S1),(S2,update_S2),(S3,update_S3),(S4,update_S4),(S5,update_S5),(S6,update_S6)]
)

get_prediction_test = theano.function(
    inputs = [inputMatrixTest,targetMatTest],
    outputs = [costTest,outputProb],
)


max_iter = 10000;
n_batch = 256
batch_sz = 256

## This is batch Stochastic Gradient Descent
for i in xrange(max_iter):
    for j in xrange(n_batch):
        Xbatch = Xval[j*batch_sz:(j*batch_sz+batch_sz),]
        Ybatch = Yval[j*batch_sz:(j*batch_sz+batch_sz),]
        
        testFine(Xbatch,Ybatch)
        if j% 10 == 0:
            cost_val, pred_val = get_prediction_test(Xval,Yval)
            print cost_val
            print pred_val
            
