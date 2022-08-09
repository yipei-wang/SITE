import torch
import random
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class_list = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
             'dog', 'frog', 'horse', 'ship', 'truck']

## functions

# show batch images
def imshow(img,
        nrow = 10,
        figsize = (10, 10),
        save = [False, None],
         cmap = 'bwr'):
    npimg = torchvision.utils.make_grid(torch.clamp(img, 0,1).cpu().detach(), nrow = nrow)
    plt.figure(figsize = figsize)
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap = cmap)
    plt.axis('off')
    plt.show()


# denormalize normalized images
def denorm(x):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    xx = x.clone().detach().to(x.device)
    xx[:, 0, :, :] = x[:, 0, :, :] * std[0] + mean[0]
    xx[:, 1, :, :] = x[:, 1, :, :] * std[1] + mean[1]
    xx[:, 2, :, :] = x[:, 2, :, :] * std[2] + mean[2]

    return xx

# normalize original images
def norm(x):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    xx = x.clone().detach().to(xdevice)
    # xx = torch.zeros(x.shape).to(device)
    xx[:, 0, :, :] = (x[:, 0, :, :] - mean[0])/std[0]
    xx[:, 1, :, :] = (x[:, 1, :, :] - mean[1])/std[1]
    xx[:, 2, :, :] = (x[:, 2, :, :] - mean[2])/std[2]

    return xx

def get_theta(batch_size, get_reverse = False):
    t = torch.rand(batch_size)*np.pi - np.pi/2
    ts = torch.rand(batch_size, 2)
    theta = torch.zeros(batch_size, 2, 3)
    theta[:,0,0] = torch.cos(t)
    theta[:,0,1] = -torch.sin(t)
    theta[:,1,0] = torch.sin(t)
    theta[:,1,1] = torch.cos(t)
    theta[:,:,2] = ts - 0.5
    
    if get_reverse:
        gamma = torch.zeros(batch_size, 2, 3)
        gamma[:,0,0] = torch.cos(-t)
        gamma[:,0,1] = -torch.sin(-t)
        gamma[:,1,0] = torch.sin(-t)
        gamma[:,1,1] = torch.cos(-t)
        gamma[:,:,2] = torch.matmul(-gamma[:,:,0:2], theta[:,:,2].view(-1,2,1)).squeeze()
    
        return theta, gamma
    return theta


def transform(image, theta, dataset = 'CIFAR'):
    bs = image.shape[0]
    if dataset == 'CIFAR':
        grid = F.affine_grid(theta, image.view(bs,3,128,128).size(), align_corners = True)
    elif dataset == 'MNIST':
        grid = F.affine_grid(theta, image.view(bs,1, 28, 28).size(), align_corners = True)
    else:
        raise AttributeError('Only accept CIFAR and MNIST datasets')
    grid = grid.float().to(image.device)
    tran = F.grid_sample(image, grid, align_corners = True)
    if dataset == 'CIFAR':
        tran[tran == 0] = tran.min()
    return tran

def transform_W(W, theta, dataset = 'MNIST'):
    if dataset == 'MNIST':
        W = W.view(-1, 1, 28, 28)
        bs = W.shape[0]
        theta_W = torch.cat([theta]*10, dim = 1).view(bs, 2, 3)
        return transform(W, theta_W, dataset = 'MNIST').view(bs//10, 10, 28, 28)

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def plot_explanation(img, explanation):
    fig, ax = plt.subplots(1,2,figsize = (6,3))
    ax[0].imshow(torch.clamp(denorm(img),0,1).squeeze().detach().cpu().numpy().transpose((1,2,0)))
    ax[0].axis('off')
    ax[1].imshow(torch.clamp(denorm(img),0,1).squeeze().detach().cpu().numpy().transpose((1,2,0)))
    ax[1].imshow(explanation.detach().cpu().numpy(), cmap = 'seismic', alpha = 0.4)
    ax[1].axis('off')
    plt.tight_layout()
    plt.show()
    
    
def plot_MNIST(img, W):
    explanation = W*img
    
    fig, ax = plt.subplots(2,11, figsize = (14,3))
    ax[0,0].imshow(img.squeeze().detach().cpu().numpy(), cmap = 'gray')
    ax[0,0].set_title('input')
    ax[1,0].imshow(img.squeeze().detach().cpu().numpy(), cmap = 'gray')
    
    for i in range(10):
        ax[0,i+1].set_title(f'{i}')
        ax[0,i+1].imshow(W[0,i].squeeze().detach().cpu().numpy(), cmap = 'bwr')
        ax[1,i+1].imshow(explanation[0,i].squeeze().detach().cpu().numpy(), cmap = 'bwr')
                 
    ax[0,10].text(30,15,'Prototypes')
    ax[1,10].text(30,15,'Explanations')
    for i in range(11):
        ax[1,i].axis('off')
        ax[0,i].axis('off')
    
    plt.tight_layout()
    plt.show()
    

def sample_prototype(prototype, label, dataset = 'MNIST'):
    batch_size = label.shape[0]
    if dataset == 'MNIST':
        target = torch.FloatTensor(batch_size, 10, 28, 28)
    elif dataset == 'CIFAR':
        target = torch.FloatTensor(batch_size, 10, 3, 128, 128)
    for i in range(10):
        x = random.sample(prototype[i], batch_size)
        if dataset == 'MNIST':
            target[:,i] = torch.cat([x[i] for i in range(batch_size)])
        elif dataset == 'CIFAR':
            target[:,i] = torch.cat([x[i].view(1,3,128,128) for i in range(batch_size)])
        
    target = target.to(label.device)
    return target

def get_prototype(train_loader, n_prototype = 1000):
    prototype = [[] for i in range(10)]
    full = [False for i in range(10)]
    for _, (image, label) in enumerate(train_loader):
        batch_size = image.shape[0]
        for i in range(batch_size):
            if not full[label[i]]:
                prototype[label[i]].append(image[i])
                if len(prototype[label[i]]) == n_prototype:
                    full[label[i]] = True
            if all(full):
                return prototype

def plot_process(image):
    x = image.squeeze().cpu().detach()
    for i in range(x.shape[0]):
        if i == 0:
            img = x[i]
        else: img = torch.cat([img, x[i]], dim = 1)
    plt.imshow(img)
    plt.axis('off')
    plt.show()