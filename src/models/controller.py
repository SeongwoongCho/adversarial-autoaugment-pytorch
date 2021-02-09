import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.autograd import Variable

NUM_OPS = 15 #16
NUM_MAGS = 10

class Controller(nn.Module):
    def __init__(self,n_subpolicies = 5, embedding_dim = 32,hidden_dim = 100):
        super(Controller, self).__init__()
        self.Q = n_subpolicies
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(NUM_OPS + NUM_MAGS,embedding_dim) # (# of operation) + (# of magnitude) 
        self.lstm = nn.LSTMCell(embedding_dim, hidden_dim)
        self.outop = nn.Linear(hidden_dim,NUM_OPS)
        self.outmag = nn.Linear(hidden_dim,NUM_MAGS)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        init_range = 0.1
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)
        self.outop.bias.data.fill_(0)
        self.outmag.bias.data.fill_(0)
        
    def get_variable(self, inputs, cuda=False, **kwargs):
        if type(inputs) in [list, np.ndarray]:
            inputs = torch.Tensor(inputs)
        if cuda:
            out = Variable(inputs.cuda(), **kwargs)
        else:
            out = Variable(inputs, **kwargs)
        return out
    
    def create_static(self,batch_size):
        inp = self.get_variable(torch.zeros(batch_size, self.embedding_dim), cuda = True, requires_grad = False)
        hx = self.get_variable(torch.zeros(batch_size, self.hidden_dim), cuda = True, requires_grad = False)
        cx = self.get_variable(torch.zeros(batch_size, self.hidden_dim), cuda = True, requires_grad = False)
        
        return inp,hx,cx
    
    def calculate(self,logits):
        probs = F.softmax(logits, dim=-1)
        log_prob = F.log_softmax(logits, dim=-1)
        entropy = -(log_prob * probs).sum(1, keepdim=False)
        action = probs.multinomial(num_samples=1).data
        selected_log_prob = log_prob.gather(1, self.get_variable(action,requires_grad = False))
        
        return entropy, selected_log_prob[:, 0], action[:,0]
    
    def forward(self,batch_size=1):
        return self.sample(batch_size)
    
    def sample(self,batch_size=1):
        policies = []
        entropies = []
        log_probs = []
           
#        inp,hx,cx = self.create_static(batch_size)
        for i in range(self.Q):
            inp,hx,cx = self.create_static(batch_size)
            for j in range(2):
#                if i > 0 or j > 0:
                if j > 0:
                    inp = self.embedding(inp) # B,embedding_dim
                hx, cx = self.lstm(inp, (hx, cx))
                op = self.outop(hx) # B,NUM_OPS
                
                entropy, log_prob, action = self.calculate(op)
                entropies.append(entropy)
                log_probs.append(log_prob)
                policies.append(action)
                
                inp = self.get_variable(action, requires_grad = False)
                inp = self.embedding(inp)
                hx, cx = self.lstm(inp, (hx, cx))
                mag = self.outmag(hx) # B,NUM_MAGS
    
                entropy, log_prob, action = self.calculate(mag)
                entropies.append(entropy)
                log_probs.append(log_prob)
                policies.append(action)
                
                inp = self.get_variable(NUM_OPS + action, requires_grad = False) 
        
        entropies = torch.stack(entropies, dim = -1) ## B,Q*4
        log_probs = torch.stack(log_probs, dim = -1) ## B,Q*4
        policies = torch.stack(policies, dim = -1) ## B,Q*4
        
        return policies, torch.sum(log_probs, dim = -1), torch.sum(entropies, dim = -1) # (B,Q*4) (B,) (B,) 