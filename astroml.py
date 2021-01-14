import csv
import numpy as np
import torch
import matplotlib.pyplot as plt


device = torch.device("cuda:0")
with open('data/stars.csv', 'r', newline='') as csvfile:
    names = csv.reader(csvfile, delimiter=',', quotechar='#', quoting=csv.QUOTE_MINIMAL)
    with open('data/lenghts.csv', 'r') as lenghts:
        lenghtreader = csv.reader(lenghts, delimiter=',', quotechar='#', quoting=csv.QUOTE_MINIMAL)
        feactures = int(next(lenghtreader)[0])
        samples = 0
        for _ in lenghtreader:
            samples += 1
    data_martix = np.zeros((samples, feactures*2, 2))
    lables_martix = np.zeros((samples,1))
    i = 0
    for name, label in names:
        data_martix[i,:,1] = label
        freq = np.loadtxt('data/{}_freq.csv'.format(name), delimiter=',')
        freq = 1/(freq)
        data_martix[i,:freq.shape[0],0] = freq
        power = np.loadtxt('data/{}_power.csv'.format(name), delimiter=',')
        data_martix[i,feactures:feactures + power.shape[0],1] = power
        
        i += 1
data_martix = data_martix.astype('float')

class Net(torch.nn.Module):

    def __init__(self, input_size):
        super(Net, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = torch.nn.Linear(input_size, 1000)
        self.fc2 = torch.nn.Linear(1000, 1000)
        self.fc3 = torch.nn.Linear(1000, 1000)
        self.fc4 = torch.nn.Linear(1000, 1000)
        self.fc5 = torch.nn.Linear(1000, 1)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        #x = torch.nn.functional.relu(self.fc4(x))
        x = self.fc5(x)
        return x

net = Net(data_martix.shape[1])
net.to(device)
criterion = torch.nn.L1Loss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
train = data_martix[int(data_martix.shape[0]*0.2):,:]
test = data_martix[:int(data_martix.shape[0]*0.2),:]
class stars_dataset(torch.utils.data.Dataset):
    def __init__(self,inputs):
        self.inputs = inputs
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = torch.tensor(self.inputs[idx,:,0])
        label = torch.tensor(self.inputs[idx,0,1])
        return {"data":data,"label":label}
    
    def __len__(self):
        return len(self.inputs)
    
dataset = stars_dataset(train)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=100,
                        shuffle=True)
losses = []
for epoch in range(10000):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs = data['data'].float().to(device)
        labels = data['label'].float().to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = torch.squeeze(net(inputs))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        print(loss.item())
        losses.append(loss.item())
torch.save(net.state_dict(), './astro_net.pth')
print("done")
plt.plot(losses[50:])
from sklearn.metrics import r2_score
output = net(torch.tensor(test[:,:,0]).float().to(device)).cpu().detach().numpy()
plt.title(r2_score(test[:,0,1],output))
plt.savefig("losses")

