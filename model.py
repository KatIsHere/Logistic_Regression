# Import useful libraries
#from sklearn.datasets import make_classification 
import matplotlib.pyplot as plt
import numpy as np
#import mnist
import tensorflow as tf
from math import sqrt
from time import time
mnist = tf.keras.datasets.mnist


# Create a class for the Softmax linear classifier
class Softmax(object):    

  def __init__(self):
    self.W = None
    self.b = None
    
  def cross_entropy(self, X, y, reg, n_features, n_samples, n_classes):
    scores = np.dot(X, self.W)+self.b
    exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))   # Normalize the scores to avoid overflowing
    probs = exp_scores/np.sum(exp_scores, axis=1, keepdims=True)
    correct_logprobs = -np.log(probs[np.arange(n_samples), y])
    loss = np.sum(correct_logprobs)/n_samples
    return loss


  def get_loss_grads(self, X, y, reg, n_features, n_samples, n_classes):
    scores = np.dot(X, self.W) + self.b
    exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))   # Normalize the scores to avoid overflowing
    probs = exp_scores/np.sum(exp_scores, axis=1, keepdims=True)          # Softmax activation
    correct_logprobs = -np.log(probs[np.arange(n_samples), y])            # Logloss of the correct class for each of our samples
    loss = np.sum(correct_logprobs)/n_samples                             # avarage loss
    reg_loss = 0.5*reg*np.sum(self.W*self.W)                              # regularization with L2 norm
    loss += reg_loss
    dscores = probs.copy() 
    dscores[np.arange(n_samples),y] -= 1        # -1 from the scores of the correct class for gradient calculation
    dscores /= n_samples

    # Gradient of the loss with respect to weights
    dW = X.T.dot(dscores) 
    # Add gradient regularization 
    dW += reg*self.W
    # Gradient of the loss with respect to biases
    db = np.sum(dscores, axis=0, keepdims=True)
    return loss, dW, db
  

  def Adam(self, X, y, alfa = 0.001, Beta1 = 0.9, Beta2 = 0.99, eps = 0.0000001, EPS = 0.00000001, reg=0.5,max_iteration = 1000):
    iteration = 0
    loss = 1.
    n_features, n_samples = X.shape[1], X.shape[0]   
    n_classes = len(np.unique(y))
    # Initialize weights from a normal distribution and the biases with zeros
    if (self.W is None) & (self.b is None):
      np.random.seed(2016) # for reproducible results
      self.W = np.random.normal(loc=0.0, scale=1e-4, size=(n_features, n_classes))
      self.b = np.zeros((1, n_classes))
    m0_W = np.zeros((n_features, n_classes))
    v0_W = np.zeros((n_features, n_classes))
    m0_b = np.zeros((1, n_classes))
    v0_b = np.zeros((1, n_classes))
    start = time()
    steps = []
    while iteration < max_iteration and loss > EPS:
        iteration+=1
        W_prev = self.W
        b_prev = self.b
        loss, dW, db = self.get_loss_grads(X, y, reg, n_features, n_samples, n_classes)
        m_W = Beta1*m0_W + (1-Beta1)*dW
        m_b = Beta1*m0_b + (1-Beta1)*db
        v_W = Beta2*v0_W + (1-Beta2)*dW*dW
        v_b = Beta2*v0_b + (1-Beta2)*db*db
        m_normed_W = m_W/(1-Beta1**iteration)
        v_normed_W = v_W/(1- Beta2**iteration)
        m_normed_b = m_b/(1-Beta1**iteration)
        v_normed_b = v_b/(1- Beta2**iteration)

        self.W -= alfa*m_normed_W/(np.sqrt(v_normed_W) + eps)
        self.b -= alfa*m_normed_b/(np.sqrt(v_normed_b) + eps)
        steps.append(self.cross_entropy(X, y, reg, n_features, n_samples, n_classes))
        v0_W = v_W
        m0_W = m_W        
        v0_b = v_b
        m0_b = m_b
    finish = time()
    return finish - start, steps

  def Nesterov_Gradient_Descent(self, X, y, gamma = 0.09, learning_rate = 0.9, EPS = 0.00000001, reg=0.5, max_iteration = 1000):
    loss = 1.
    iteration = 0
    n_features, n_samples = X.shape[1], X.shape[0]   
    n_classes = len(np.unique(y))
    # Initialize weights from a normal distribution and the biases with zeros
    if (self.W is None) & (self.b is None):
      np.random.seed(2016) # for reproducible results
      self.W = np.random.normal(loc=0.0, scale=1e-4, size=(n_features, n_classes))
      self.b = np.zeros((1, n_classes))
      
    y0_W = self.W
    y0_b = self.b
    l_prev = 0
    l = (1 + sqrt(1 + 4*l_prev**2))*0.5
    L = 10
    start = time()
    steps = []
    while iteration < max_iteration and loss > EPS:
        steps.append(self.cross_entropy(X, y, reg, n_features, n_samples, n_classes))
        W_prev = self.W
        b_prev = self.b
        loss, dW, db = self.get_loss_grads(X, y, reg, n_features, n_samples, n_classes)
        
        self.W = y0_W - 1/L * dW
        y0_W = self.W + (l_prev - 1)/l *(self.W - W_prev)

        self.b = y0_b + 1/L * db
        y0_b = self.b + (l_prev - 1)/l *(self.b - b_prev) 

        l_prev = l
        l = (1 + sqrt(1 + 4*l_prev**2))*0.5
        iteration+=1
    finish = time()
    return finish - start, steps

  
  def Conditional_Gradient_Descent(self, X, y, EPS = 0.00000001, reg=0.5, max_iteration = 1000):
    loss = 1.
    iteration = 0
    n_features, n_samples = X.shape[1], X.shape[0]   
    n_classes = len(np.unique(y))
    # Initialize weights from a normal distribution and the biases with zeros
    if (self.W is None) & (self.b is None):
      np.random.seed(2016) # for reproducible results
      self.W = np.random.normal(loc=0.0, scale=1e-4, size=(n_features, n_classes))
      self.b = np.zeros((1, n_classes))
    start = time()
    steps = []
    while iteration < max_iteration and loss > EPS:
        iteration+=1
        loss, dW, db = self.get_loss_grads(X, y, reg, n_features, n_samples, n_classes)
        W_prev = self.W
        b_prev = self.b
        if np.argmin(W_prev) > np.argmin(dW):
          y_W = W_prev
        else:
          y_W = dW        
        if np.argmin(b_prev) > np.argmin(db):
          y_b = b_prev
        else:
          y_b = db
        k = 2/(iteration + 2)
        self.W = k * y_W + (1 - k) * W_prev
        self.b = k * y_b + (1 - k) * b_prev
        steps.append(self.cross_entropy(X, y, reg, n_features, n_samples, n_classes))
    finish = time()
    return finish - start

  def Gradient_Descent(self, X, y, learning_rate=1e-4, EPS = 0.00000001, reg=0.5, max_iteration=1000):
    # Get useful parameters
    n_features, n_samples = X.shape[1], X.shape[0]   
    n_classes = len(np.unique(y))
    
    # Initialize weights from a normal distribution and the biases with zeros
    if (self.W is None) & (self.b is None):
      np.random.seed(2016) # for reproducible results
      self.W = np.random.normal(loc=0.0, scale=1e-4, size=(n_features, n_classes))
      self.b = np.zeros((1, n_classes))
    iteration = 0
    loss = 1.
    start = time()
    steps = []
    while iteration < max_iteration and loss > EPS:
      # Get loss and gradients
      loss, dW, db = self.get_loss_grads(X, y, reg, n_features, n_samples, n_classes)
      iteration += 1
      # update weights and biases
      self.W -= learning_rate*dW
      self.b -= learning_rate*db
      steps.append(self.cross_entropy(X, y, reg, n_features, n_samples, n_classes))
    finish = time()
    return finish - start, steps
          
  def predict(self, X):
    
    y_pred = np.dot(X, self.W)+self.b
    y_pred=np.argmax(y_pred, axis=1)
    
    return y_pred


(X_Train, y_Train),(X_test, y_test) = mnist.load_data()
# Getting some useful parameters
n_size = 28
n_samples = X_Train.shape[0] # number of samples
n_classes = len(np.unique(y_Train)) # number of classes in the dataset
n_test_samples = X_test.shape[0]

X_Train = np.reshape(X_Train, (n_samples, n_size*n_size))
X_test = np.reshape(X_test, (n_test_samples, n_size*n_size))
n_features = X_Train.shape[1] # number of features 
X_Train, x_test = X_Train / 255.0, X_test / 255.0

# Split dataset into training and validation
#X_train, y_train = X[0:800], y[0:800]
X_val, y_val = X_test, y_test

time_Gradient = 0
time_Nesterow = 0
time_Adam = 0
TRAIN_ACCURACY_Gradient = []
TEST_ACCURACY_Gradient = []
TRAIN_ACCURACY_Nesterow = []
TEST_ACCURACY_Nesterow = []
TRAIN_ACCURACY_Adam = []
TEST_ACCURACY_Adam = []
for i in range(1):
  # Train on the entire dataset
  X_train = X_Train[:100, :]
  y_train = y_Train[:100]
  softmax = Softmax()
  time_passed, steps_Gr = softmax.Gradient_Descent(X_train, y_train, learning_rate=0.09, reg=0.01, EPS = 0.1, max_iteration=200)
  acc_train = np.mean(softmax.predict(X_train)==y_train)
  acc_test = np.mean(softmax.predict(X_val)==y_val)
  time_Gradient += time_passed
  #print('Gradient_Descent Training accuracy', acc_train)
  #print('Gradient_Descent Validation accuracy', acc_test)
  TRAIN_ACCURACY_Gradient.append(acc_train)
  TEST_ACCURACY_Gradient.append(acc_test)
  #print("Execution time: ", time_passed)
  #Plot(X_train, y_train, softmax, "gradient descent")
  #print()
  # Train on the entire dataset
  softmax = Softmax()
  time_passed, steps_Nest = softmax.Nesterov_Gradient_Descent(X_train, y_train, learning_rate=0.9, reg=0.01, EPS = 0.1, max_iteration=200)
  acc_train = np.mean(softmax.predict(X_train)==y_train)
  acc_test = np.mean(softmax.predict(X_val)==y_val)
  time_Nesterow += time_passed
  #print('Nesterov_Gradient_Descent Training accuracy', acc_train)
  #print('Nesterov_Gradient_Descent Validation accuracy',np.mean(softmax.predict(X_val)==y_val))
  TRAIN_ACCURACY_Nesterow.append(acc_train)
  TEST_ACCURACY_Nesterow.append(acc_test)
  #print("Execution time: ", time_passed)
  #Plot(X_train, y_train, softmax, "Nesterov")
  #print()
  # Train on the entire dataset

  softmax = Softmax()
  time_passed, steps_Ad = softmax.Adam(X_train, y_train,reg=0.01, EPS = 0.1, max_iteration=200)
  acc_train = np.mean(softmax.predict(X_train)==y_train)
  acc_test = np.mean(softmax.predict(X_val)==y_val)
  time_Adam += time_passed
  #print('Adam Training accuracy', acc_train)
  #print('Adam Validation accuracy', acc_train)
  TRAIN_ACCURACY_Adam.append(acc_train)
  TEST_ACCURACY_Adam.append(acc_test)
  #print("Execution time: ", time_passed)
  #Plot(X_train, y_train, softmax, "Adam")
  #print()
  #softmax = Softmax()
  #time_passed, steps_Ad = softmax.Conditional_Gradient_Descent(X_train, y_train, reg=0.01, EPS = 0.1, max_iteration=200)
  #acc_train = np.mean(softmax.predict(X_train)==y_train)
  #acc_test = np.mean(softmax.predict(X_val)==y_val)
  #TRAIN_ACCURACY_Adam.append(acc_train)
  #TEST_ACCURACY_Adam.append(acc_test)

gr_s = np.linspace(0, 1, len(steps_Gr))
Nest_s = np.linspace(0, 1, len(steps_Nest))
Ad_s = np.linspace(0, 1, len(steps_Ad))
plt.plot(gr_s, steps_Gr,'o-', label = "Grad")
plt.plot(Nest_s, steps_Nest, 'o-', label = "Nesterow")
plt.plot(Ad_s, steps_Ad, 'o-', label = "Adam")
# # Train on the entire dataset
# start = time.time()
# softmax = Softmax()
# softmax.Conditional_Gradient_Descent(X_train, y_train, reg=0.1, EPS = 0.0001, max_iteration=1000000)
# finish = time.time()
# print('Training accuracy', np.mean(softmax.predict(X_train)==y_train))
# print('Validation accuracy',np.mean(softmax.predict(X_val)==y_val))
# print("Execution time: ", finish - start)
print("Gradient time passed = ", time_Gradient, "\n Nesterow time passed = ", time_Nesterow, "\n Adam time passed = ", time_Adam)
#x_vals = np.linspace(500, X_Train.shape[0], X_Train.shape[0]/100 - 5)
#plt.plot(x_vals, TRAIN_ACCURACY_Gradient, label = ("Train Gradient, time passed : " + str(time_Gradient) + "sec"))
#plt.plot(x_vals, TEST_ACCURACY_Gradient, label = "Test gradient")
#plt.plot(x_vals, TRAIN_ACCURACY_Nesterow, label = ("Train Nesterow, time passed : " + str(time_Nesterow) + "sec"))
#plt.plot(x_vals, TEST_ACCURACY_Nesterow, label = "Test Nesterow")
#plt.plot(x_vals, TRAIN_ACCURACY_Adam, label = ("Train Adam, time_passed : " + str(time_Adam) + "sec"))
#plt.plot(x_vals, TEST_ACCURACY_Adam, label = "Test Adam")
plt.legend()
plt.show()