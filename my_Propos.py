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

#k-means,get the cluster assignment posterior probability p(k|x)
def k_means(x, n_clusters, centroids):
    N, D = x.size()
    #repeat the centroids to have same size as x
    c = centroids

    #compute the distances between each sample and each centroid
    distances = torch.cdist(x, c)
    #compute the cluster assignment postrerior probability p(k|x)
    labels = torch.argmin(distances, dim=1)
    p_k_x = torch.zeros(N, n_clusters)
    p_k_x[torch.arange(N), labels] = 1
    return p_k_x


    

        
   


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

# exponential moving average
class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


#Propos
#hyperparameters
batch_size = 128
moving_average_decay = 0.99
learning_rate = 0.05
num_epochs = 100



#load the dataset
datasets = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())


#load the dataloader
dataloader = torch.utils.data.DataLoader(datasets, batch_size=batch_size, shuffle=True, num_workers=2)

#initialize the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#initialize the online network and target network
online_net = Online_Net().to(device)
target_net = None


#initialize the updater for the target network
target_ema_updater = EMA(moving_average_decay)



#initialize the projection head
online_predictor = MLP(1000, 1000).to(device)


#initialize the optimizer
optimizer_online = torch.optim.SGD(online_net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
optimizer_predictor = torch.optim.SGD(online_predictor.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)


#initialize the augmentation
augmentation = Default_Augmentation(32).to(device)


#randomly initialize the centroids
centroids = torch.randn(1000, 10).to(device)

for epoch in range(num_epochs):
    
    for i, mini_batch in enumerate(dataloader):
        images, _ = mini_batch
        images = images.to(device)


        target_features = target_net(images)
        p_k_x = k_means(target_features, 10, centroids=centroids)
        #get the augmented images
        images_one, images_two = augmentation(images)

        
    
    
        
        



