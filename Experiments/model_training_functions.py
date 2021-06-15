import math, random, torch, collections, time, torch.nn.functional as F, networkx as nx, matplotlib.pyplot as plt, numpy as np

from torch import Tensor
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, matmul, fill_diag, sum, mul
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from IPython.display import clear_output
from torch_geometric.utils import to_networkx
from functools import wraps

from helper import *


def train_one_epoch(model, criterion, optimizer, x, y, train_mask = None,
                    reg=False): #x is a dictionary
    model.train()
    out = model(**x)
    loss = criterion(out, y) if train_mask is None else criterion(out[train_mask], y[train_mask])
    if reg:
      predicted = out.detach()
    else:
      _, predicted = torch.max(out.detach(),1)
    optimizer.zero_grad()  
    loss.backward()  
    optimizer.step()
    
    model.eval()
    with torch.no_grad():
        if train_mask is None:
            length = len(y)
            if reg == True:
              accuracy = torch.mean((predicted - y)**2).item()/length
              misclassified = None
            else:
              accuracy = (predicted == y).sum().item()/length
              misclassified = None
        else:
            if reg == True:
                length = len(y[train_mask])
                accuracy = torch.sum((predicted[train_mask] - y[train_mask])**2).item()/length
                misclassified = None
            else:
                length = len(y[train_mask])
                accuracy = (predicted[train_mask] == y[train_mask].detach()).sum().item()/length
                misclassified = (predicted[train_mask] != y[train_mask]).numpy()
    
    return out, loss.item(), accuracy, misclassified

def test(model, x, y, test_mask = None, reg=False): #x is a dictionary
    model.eval()
    with torch.no_grad():
        out = model(**x)
        #print(out)
        if reg == True:
          predicted = out.detach()
        else:
          _, predicted = torch.max(out.detach(), 1)
        if test_mask is None:
            length = len(y)
            if reg == True:
              accuracy = torch.sqrt(torch.nn.MSELoss()(predicted[test_mask], y).detach())
              misclassified = None
            else:
              accuracy = (predicted == y).sum().item()/length
              misclassified = None
        else:
            if reg == True:
                length = len(y[test_mask])
                accuracy = torch.sqrt(torch.nn.MSELoss()(predicted[test_mask], y[test_mask]).detach())
                misclassified = None
            else:
                length = len(y[test_mask])
                accuracy = (predicted[test_mask] == y[test_mask].detach()).sum().item()/length
                misclassified = (predicted[test_mask] != y[test_mask]).numpy()
    return accuracy, predicted[test_mask].numpy()
    
def plot_acc(train_acc, test_acc=None, xaxis = 'epochs', yaxis = 'accuracy', title = 'Accuracy plot'):
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    if test_acc is not None:
        plt.plot(np.arange(len(train_acc)), train_acc, color='red')
        plt.plot(np.arange(len(test_acc)), test_acc, color='blue')
        plt.legend(['train accuracy', 'test accuracy'], loc='upper right')
    else: 
        plt.plot(np.arange(len(train_acc)), train_acc, color='red')
        plt.legend(['train accuracy'], loc='upper right')
    plt.title(title)
    plt.tight_layout()
    plt.show() #show train_acc and test_acc together
    
def plot_loss(loss, xaxis = 'epochs', yaxis = 'loss', title = 'Loss plot'):
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    plt.plot(np.arange(len(loss)), loss, color='black')
    plt.title(title)
    plt.tight_layout()
    plt.show()

@timethis
def train(epochs, model, criterion, optimizer, x, y, m = mask(None, None), 
          plotting = True, scatter_size = 30, plotting_freq = 50, dim_reduction = 'pca',
          reg=False):
    dim_reduction_dict = {'pca': visualize_pca, 'tsne': visualize_tsne}
    train_acc_list = []
    test_acc_list = []
    loss_list = []
    for epoch in range(epochs):
        out, loss, train_acc, misclassified = train_one_epoch(model, criterion, optimizer, x, y, m.train,
                                                              reg=reg)
        model.eval()
        test_acc, predictions = test(model, x, y, m.test, reg=reg)
        #print(model(x['x']))
        train_acc_list.append(train_acc)
        loss_list.append(loss)
        test_acc_list.append(test_acc)
        if epoch % plotting_freq == 0: print("It: {:.1f}, Loss: {:.4f}, Train/Test accuracy: {:.4f}/{:.4f}".format( epoch, loss, train_acc_list[-1]/float(torch.sum( m.train).item()), 
                                                                                                                   test_acc_list[-1]/float(torch.sum( m.test).item())))
        if plotting and ~reg:
            print("here")
            if epoch % plotting_freq == 0:
                print("here 2")
                clear_output(wait=True)
                dim_reduction_dict[dim_reduction](out, color=y, size = scatter_size, epoch=epoch, loss = loss)
    if plotting:
        if m == mask(None, None):
            plot_acc(train_acc_list)
        else:
            plot_acc(train_acc_list, test_acc_list)
        plot_loss(loss_list)
    print("Final test accuracy: {:.2f}".format( test_acc_list[-1]))
    return train_acc_list, test_acc_list, loss_list, misclassified, predictions
