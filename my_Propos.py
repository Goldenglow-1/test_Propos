#Bootstrap your own latent
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

#Feature extraction
class Online_Net(nn.Module):
    def __init__(self):
        super(Online_Net, self).__init__()
        #backbone
        self.model = torchvision.models.resnet18(pretrained=False)
        #fit with the CIFAR10(small size)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=3, bias=False)
        self.model.maxpool=nn.Identity()

    def forward(self, x):
        output = self.model(x)
        #size of the ouput is (batch_size,1000)
        return output


class Target_Net(nn.Module):
    def __init__(self):
        super(Online_Net, self).__init__()
        self.model = torchvision.models.resnet18(pretrained=False)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=3, bias=False)
        self.model.maxpool=nn.Identity()
    def forward(self, x):
        output = self.model(x)
        #size of the ouput is (batch_size,1000)
        return output


#Positive_Sampling
class Positive_Sampling(nn.Module):
    #v = f(x) + σ * ε, ε ~ N(0,1)
    def __init__(self, variance):
        super(Positive_Sampling, self).__init__()
        self.variance = variance
    
    def forward(self, x):
        #x.size() = (batch_size,1000)
        #output.size() = (batch_size,1000)
        output = x + self.variance * torch.randn(x.size())
        return output

#MLP
def MLP(projection_size, hidden_size):
    return nn.Sequential(
        nn.Linear(projection_size, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size),
    )

#k-means
def k_means(x, n_clusters):
    N, D = x.size()
    #randomly choose n_clusters samples as the initial centroids
    c = x[torch.randperm(N)[:n_clusters]]
    #repeat the centroids to have same size as x
    c = torch.repeat_interleave(c, repeats=N//n_clusters, dim=0)
    #add the remaining centroids
    c = torch.cat([c, x[:N % n_clusters]], dim=0)
    #repeat the x to have same size as centroids
    x = torch.repeat_interleave(x, repeats=n_clusters, dim=0)
    #assign the labels
    labels = torch.zeros(N, dtype=torch.long)
    
    #compute the distances between each sample and each centroid
    distances = torch.cdist(x, c)
    #assign the labels
    new_labels = torch.argmin(distances, dim=1)
    #if the labels are the same, then break
        
    #assign the new labels
    labels = new_labels
    #update the centroids
    for i in range(n_clusters):
        c[i] = x[labels == i].mean(dim=0)
    return labels, c


#augmentation utils——SimCLR
def default(val, def_val):
    return def_val if val is None else val

class Default_Augmentation(nn.Module):
    def __init__(self, image_size, augment_fn=None, augment_fn2=None):
        super(Default_Augmentation, self).__init__()
        DEFAULT_AUG = torch.nn.Sequential(
            RandomApply(
                transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
                p = 0.3
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop((image_size, image_size)),
            transforms.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225])),
        )
        self.augment1 = default(augment_fn, DEFAULT_AUG)
        self.augment2 = default(augment_fn2, self.augment1)


    
    def forward(self, x):
        image_one, image_two = self.augment1(x), self.augment2(x)
        return image_one, image_two
        
    


class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)

# default SimCLR augmentation



#Propos
#load the datasets
datasets = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
#load the dataloader
batch_size = 128
dataloader = torch.utils.data.DataLoader(datasets, batch_size=batch_size, shuffle=True, num_workers=2)
for data in dataloader:
    images, labels = data
    print(images.size())
    print(labels.size())
    break




