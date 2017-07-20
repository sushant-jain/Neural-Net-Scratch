
# coding: utf-8

# In[2]:


import numpy as np


# In[3]:


MNIST_TRAIN_DIR="/home/sushant/ml/neuralScratch/MNIST/train.csv"

MNIST_TEST_DIR="/home/sushant/ml/neuralScratch/MNIST/test.csv"


# In[4]:


data=np.genfromtxt(MNIST_TRAIN_DIR,delimiter=',')
data=data[1:,:]

Y=data[:,0]
X=data[:,1:]

Y=Y.astype(np.int64)

testdata=np.genfromtxt(MNIST_TEST_DIR,delimiter=',')


# In[5]:


Y_train=Y[:int(0.8*len(Y))]
X_train=X[:int(0.8*len(X)),:]

Y_validate=Y[int(0.8*len(Y)):]
X_validate=X[int(0.8*len(Y)):,:]


# In[6]:


# Y_train=Y_train[:1000]
# X_train=X_train[:1000]


# In[7]:


num_examples = len(X_train) # training set size
nn_input_dim = 784 # input layer dimensionality
nn_output_dim = 10 # output layer dimensionality
 
# Gradient descent parameters (I picked these by hand)
epsilon = 0.0000001 # learning rate for gradient descent
reg_lambda = 0.00001 # regularization strength


# In[18]:


#helper function to forward propogate
def forwardProp(model,input_data):
    W1, b1, W2, b2,W3, b3,W4,b4 = model['W1'], model['b1'], model['W2'], model['b2'],model['W3'],model['b3'],model['W4'],model['b4']

    # Forward propagation to calculate our predictions
    z1 = input_data.dot(W1) + b1
    a1 = np.tanh(z1)
    # a1=z1
    # a1[a1<0]=0
    z2 = a1.dot(W2) + b2
    a2=np.tanh(z2)
    # a2=z2
    # a2[a2<0]=0
    z3=a2.dot(W3)+b3
    a3=np.tanh(z3)
    # a3=z3
    # a3[a3<0]=0
    z4=a3.dot(W4)+b4

    #print "z4",z4[:,:]

    az4=np.absolute(z4)
    
    #print "az4",az4[:,:]

    #print "sum",np.sum(az4,axis=1,keepdims=True)

    exp_scores = np.exp(z4/np.sum(az4,axis=1,keepdims=True))

    #print "e",exp_scores[:,:]

    # print "expFP"
    # print exp_scores[:,:]
    # print "z4"
    # print z4[:,:]


    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    return probs


# In[ ]:





# In[9]:


# Helper function to evaluate the total loss on the dataset
def calculate_loss(model,input_data,label):
    W1, b1, W2, b2,W3, b3,W4,b4 = model['W1'], model['b1'], model['W2'], model['b2'],model['W3'],model['b3'],model['W4'],model['b4']

    probs=forwardProp(model,input_data)
    
    # Calculating the loss
    corect_logprobs = -np.log(probs[range(len(input_data)), label])
    data_loss = np.sum(corect_logprobs)
    # Add regulatization term to loss (optional)
    data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2))+np.sum(np.square(W3)) + np.sum(np.square(W4)))
    return 1./len(input_data) * data_loss


# In[10]:


# Helper function to predict an output (0 or 1)
def predict(model, x):
    probs=forwardProp(model,x)  
    #print probs[:,:] 
    return np.argmax(probs, axis=1)


# In[21]:


# This function learns parameters for the neural network and returns the model.
# - nn_hdim: Number of nodes in the hidden layer
# - num_passes: Number of passes through the training data for gradient descent
# - print_loss: If True, print the loss every 1000 iterations
def build_model(nn_hdim, num_passes=20000, print_loss=False,model={}):
     
    # Initialize the parameters to random values. We need to learn these.
    if model=={}:
	    np.random.seed(0)
	    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
	    b1 = np.zeros((1, nn_hdim))
	    
	    W2 = np.random.randn(nn_hdim, nn_hdim) / np.sqrt(nn_hdim)
	    b2 = np.zeros((1, nn_hdim))
	    
	    W3 = np.random.randn(nn_hdim, nn_hdim) / np.sqrt(nn_hdim)
	    b3 = np.zeros((1, nn_hdim))
	    
	    W4 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
	    b4 = np.zeros((1, nn_output_dim))
	 
	    # This is what we return at the end
	    model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2,'W3':W3,'b3':b3,'W4':W4,'b4':b4}
    else:
    	W1, b1, W2, b2,W3, b3,W4,b4 = model['W1'], model['b1'], model['W2'], model['b2'],model['W3'],model['b3'],model['W4'],model['b4']


    epsilon = 0.0000001 # learning rate for gradient descent
     
    # Gradient descent. For each batch...
    for i in xrange(1, num_passes):
 
        if(i%25==0):
        	epsilon*=0.7
        # Forward propagation
        z1 = X_train.dot(W1) + b1
      	a1 = np.tanh(z1)
        # a1=z1
        # a1[a1<0]=0
        z2 = a1.dot(W2) + b2
       	a2=np.tanh(z2)
        # a2=z2
        # a2[a2<0]=0
        z3=a2.dot(W3)+b3
      	a3=np.tanh(z3)
        # a3=z3
        # a3[a3<0]=0
 # print "after back prop"
        # print "w1",W1[:,:]
        # print "w2",W2[:,:]
        # print  "w3",W3[:,:]
        # print "w4",W4[:,:]
        # print "a3",a3[:,:]
        # print "w4",W4[:,:]
        # print "b4",b4[:,:]

        z4=a3.dot(W4)+b4

	#print "z4",z4[:,:]    
        az4=np.absolute(z4)
    
    	exp_scores = np.exp(z4/np.sum(az4,axis=1,keepdims=True))
        
        # print "exp"
        # print exp_scores[:,:]
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
 
        # Backpropagation
        delta5 = probs
        delta5[range(num_examples), Y_train] -= 1
        
        dW4=(a3.T).dot(delta5)
        db4=np.sum(delta5,axis=0,keepdims=True)
        
       	delta4=delta5.dot(W4.T)*(1-np.power(a3,2))
        # a3[a3>0]=1
        # delta4=delta5.dot(W4.T)*a3
        
        dW3=(a2.T).dot(delta4)
        db3=np.sum(delta4,axis=0,keepdims=True)
        
        delta3=delta4.dot(W3.T)*(1-np.power(a2,2))
        # a2[a2>0]=1
        # delta3=delta4.dot(W3.T)*a2
        
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        
       	delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        # a1[a1>0]=1
        # delta2=delta3.dot(W2.T)*a1
        
        dW1 = (X_train.T).dot(delta2)
        db1 = np.sum(delta2, axis=0)/nn_hdim
        
        # if i%10==0:
        #     print "delta5"
        #     print delta5[0,:]
            
            
 
        # Add regularization terms (b1 and b2 don't have regularization terms)
        dW4 += reg_lambda * W4
        dW3+=reg_lambda*W3
        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1

        # print "before back prop"
        # print "w1",W1[:,:]
        # print "w2",W2[:,:]
        # print  "w3",W3[:,:]
        # print "w4",W4[:,:]
 
        # Gradient descent parameter update
        W1 -= epsilon * dW1
        b1 -= epsilon * db1
        W2 -= epsilon * dW2
        b2 -= epsilon * db2
        W3 -= epsilon*dW3
        b3 -= epsilon*db3
        W4 -= epsilon*dW4
        b4 -= epsilon*db4
         

        # print "after back prop"
        # print "w1",W1[:,:]
        # print "w2",W2[:,:]
        # print  "w3",W3[:,:]
        # print "w4",W4[:,:]
        # Assign new parameters to the model
        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2,'W3':W3,'b3':b3,'W4':W4,'b4':b4}
         
        # Optionally print the loss.
        # This is expensive because it uses the whole dataset, so we don't want to do it too often.
        if print_loss and i % 10 == 0:
          print "Loss after iteration %i: Training Loss: %f    Validation Loss: %f"%(i, calculate_loss(model,X_train,Y_train),calculate_loss(model,X_validate,Y_validate))
     
    return model


# In[ ]:

model=np.load("model_1e7_decay.npy").item()
model = build_model(500,num_passes=100, print_loss=True,model=model)


predictions=predict(model,testdata)

print predictions[:]

np.save("predictions.npy",predictions)
np.save("model_1e7_decay2.npy",model)

validation_prediction=predict(model,X_validate)

count=0;

print validation_prediction[:]
print Y_validate[:]

for i in range(len(Y_validate)):
	if validation_prediction[i]==Y_validate[i]:
		count+=1

print "validation accuracy",float(count/float(len(Y_validate)))


# In[ ]:





# In[ ]:


