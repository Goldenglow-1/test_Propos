#Bootstrap your own latent
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

#Feature extraction
class Online_Net(nn.Module):
    def __init__(self):
        super(Online_Net, self).__init__()
        self.model = torchvision.models.resnet18(pretrained=False)
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


#Propos
class Propos(nn.Module):
    def __init__(self, projection_size, hidden_size, n_clusters, variance):
        super(Propos, self).__init__()
        self.online_net = Online_Net()
        self.target_net = Target_Net()
        self.positive_sampling = Positive_Sampling(variance)
        self.projection_size = projection_size
        self.hidden_size = hidden_size
        self.n_clusters = n_clusters
        self.predictor = MLP(projection_size, hidden_size)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, x):
        #x.size() = (batch_size,3,224,224)
        #output.size() = (batch_size,1000)
        output = self.online_net(x)
        return output
    
    def update_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())
    
    def update_predictor(self, x):
        #x.size() = (batch_size,1000)
        #output.size() = (batch_size,projection_size)
        output = self.predictor(x)
        return output
    
    def update_labels(self, x):
        #x.size() = (batch_size,projection_size)
        #labels.size() = (batch_size)
        labels, _ = k_means(x, self.n_clusters)
        return labels
    
    def update_loss(self, x, labels):
        #x.size() = (batch_size,projection_size)
        #labels.size() = (batch_size)
        #loss.size() = (1)
        loss = self.criterion(x, labels)
        return loss
    
    def update(self, x):
        #x.size() = (batch_size,3,224,224)
        #output.size() = (batch_size,1000)
        output = self.forward(x)
        #output.size() = (batch_size,projection_size)
        output = self.update_mlp(output)
        #labels.size() = (batch_size)
        labels = self.update_labels(output)
        #loss.size() = (1)
        loss = self.update_loss(output, labels)
        #update the parameters
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss
    
    def update_online_net(self, x):
        #x.size() = (batch_size,3,224,224)
        #output.size() = (batch_size,1000)
        output = self.forward(x)
        #output.size() = (batch_size,projection_size)
        output = self.update_mlp(output)
        #labels.size() = (batch_size)
        labels = self.update_labels(output)
        #loss.size() = (1)
        loss = self.update_loss(output, labels)
        #update the parameters
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss
    
    def update_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())
