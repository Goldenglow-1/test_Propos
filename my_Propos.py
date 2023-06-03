#Bootstrap your own latent
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import copy
from matplotlib import pyplot as plt


def plot_curve(data):#visualize the loss function
    fig = plt.figure()
    plt.plot(range(len(data)), data, color = 'blue')
    plt.legend(['value'],loc = 'upper right')
    plt.xlabel('step')
    plt.ylabel('value')
    plt.show()



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


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

def get_target_net(online_net):
    target_net = copy.deepcopy(online_net)
    set_requires_grad(target_net, False)
    return target_net
# class Target_Net(nn.Module):
#     def __init__(self):
#         super(Online_Net, self).__init__()
#         self.model = torchvision.models.resnet18(pretrained=False)
#         self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=3, bias=False)
#         self.model.maxpool=nn.Identity()
#     def forward(self, x):
#         output = self.model(x)
#         #size of the ouput is (batch_size,1000)
#         return output


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)



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
def get_p_k_x(x, n_clusters, centroids):
    N, D = x.size()
    
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


#define the cluster loss
class cluster_loss(nn.Module):
    def __init__(self, n_clusters, temperature):
        super(cluster_loss, self).__init__()
        self.n_clusters = n_clusters
        self.temperature = temperature

    def forward(self, centers, centers_prime):

        mat = torch.matmul(centers, centers_prime.T)

        part_1 = -mat.diag().mean()/self.temperature.float()

        mat = torch.exp(mat/ self.temperature)

        mat = mat.fill_diagonal_(0)

        part_2 = torch.log(mat.sum(dim=1, keepdim=True)).mean().float()

        loss = part_1 + part_2
        
        return loss



#update the centroids in CIFAR10 datasets
def update_centroids(target_features_from_raw_input, p_k_x, num_clusters):
    

    #data.shape = (N, D)
    centroids = torch.empty((num_clusters, target_features_from_raw_input.shape[1]))
    for i in range(num_clusters):

        #p_k_x.shape = (N,num_clusters)
        #get the points that belong to the i-th cluster
        cluster_points = target_features_from_raw_input[p_k_x[:, i] == 1]
        #compute the mean of the points
        centroids[i] = torch.mean(cluster_points, dim=0)
    return centroids




#Propos
#hyperparameters
batch_size = 128
moving_average_decay = 0.99
learning_rate = 0.05
num_epochs = 100
n_clusters = 10
temperature = 0.5
variance = 0.001
lamda_PSL = 0.1

#load the dataset
datasets = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())


#load the dataloader
dataloader = torch.utils.data.DataLoader(datasets, batch_size=batch_size, shuffle=True, num_workers=2)


#get all the images from the dataset
images_all = torch.cat([x[0] for x in dataloader], dim=0)


#initialize the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#initialize the online network and target network
online_net = Online_Net().to(device)



target_net = None


#initialize the updater for the target network
target_ema_updater = EMA(moving_average_decay)

target_moving_average = update_moving_average(target_ema_updater, target_net, online_net)

#initialize the projection head
online_predictor = MLP(1000, 1000).to(device)


#initialize the optimizer
optimizer_online = torch.optim.SGD(online_net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
optimizer_predictor = torch.optim.SGD(online_predictor.parameters(), lr=10*learning_rate, momentum=0.9, weight_decay=1e-4)

#initialize the cluster loss
Cluster_Loss = cluster_loss(n_clusters, temperature=temperature).to(device)


#initialize the augmentation
augmentation = Default_Augmentation(32).to(device)


#randomly initialize the centroids
centroids = torch.randn(n_clusters, 1000).to(device)



train_loss= []


for epoch in range(num_epochs):



    with torch.no_grad():
        target_net = get_target_net(online_net)

    #update the centroids and the p_k_x
    target_features_from_raw_input_all = target_net(images_all)
    p_k_x_all = get_p_k_x(target_features_from_raw_input_all, n_clusters, centroids=centroids)
    centroids = update_centroids(target_features_from_raw_input_all, p_k_x_all, n_clusters)



    for i, mini_batch in enumerate(dataloader):
            images, _ = mini_batch
            images = images.to(device)


            target_features_from_raw_input = target_net(images)
            p_k_x_batch_size = get_p_k_x(target_features_from_raw_input, 10, centroids=centroids)
            #get the augmented images
            images_one, images_two = augmentation(images)
            #get the online features
            online_features = online_net(images_one)
            target_features = target_net(images_two)


            #get the positive samples
            positive_samples = Positive_Sampling(variance=variance).to(device)

            #get the positive features
            positive_features = positive_samples(online_features)

            #compute the cluster centers for the online features
            centroids_online_features = torch.matmul(p_k_x_batch_size.T, online_features)

            centroids_online_features = centroids_online_features / torch.norm(centroids_online_features, dim=1, keepdim=True)

            #the size of centroids_online_features is (n_clusters,1000)


            #compute the cluster centers for the target features
            centroids_target_features = torch.matmul(p_k_x_batch_size.T, target_features)

            centroids_target_features = centroids_target_features / torch.norm(centroids_target_features, dim=1, keepdim=True)

            #the size of centroids_target_features is (n_clusters,1000)


            #get the predeicted features
            predicted_features = online_predictor(positive_features)


            optimizer_online.zero_grad()
            optimizer_predictor.zero_grad()

            #computer the loss
            loss_PSA = torch.norm(predicted_features - target_features, dim=1).mean()


            loss_PSL = Cluster_Loss(centroids_online_features, centroids_target_features)

            loss = loss_PSA + lamda_PSL * loss_PSL

            loss.backward()

            target_moving_average()
            optimizer_online.step()
            optimizer_predictor.step()
            train_loss.append(loss.item())
            if i % 50 == 0:
                print('epoch:{}, iter:{}, loss:{}'.format(epoch, i, loss.item()))

plot_curve(train_loss, 'train_loss')
torch.save(online_net.state_dict(), 'online_net.pth')


        

        

        
    
    
        
        



