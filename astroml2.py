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
    data_martix = np.zeros((samples * 21, 60, 2))
    lables_martix = np.zeros((samples * 21,1))
    i = 0
    for name, label in names:
        freq = np.loadtxt('data/{}_freq.csv'.format(name), delimiter=',')
        freq = 1/(freq)
        label = int(float(label))
        if label == -1:
            indexes = np.random.choice(freq.shape[0] - 30, 20, replace = False)
            power = np.loadtxt('data/{}_power.csv'.format(name), delimiter=',')
            for x in indexes: 
                data_martix[i,:,0] = np.array(list(freq[x:x + 30]) + list(power[x:x + 30]))
                data_martix[i,:,1] = (x <= label) and (label < (x + 30)) 
                i += 1
        else:
            if freq.shape[0] < label:
                continue
            indexes = np.random.choice(freq.shape[0] - 30, 20, replace = False)
            power = np.loadtxt('data/{}_power.csv'.format(name), delimiter=',')
            for x in indexes: 
                data_martix[i,:,0] = np.array(list(freq[x:x + 30]) + list(power[x:x + 30]))
                data_martix[i,:,1] = (x <= label) and (label < (x + 30)) 
                i += 1
            if label + 30 > freq.shape[0]:
                label -= (label + 30) - freq.shape[0] 
            data_martix[i,:,0] = np.array(list(freq[label:label + 30]) + list(power[label:label + 30]))
            data_martix[i,:,1] = 1
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
        self.fc5 = torch.nn.Linear(1000, 2)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        #x = torch.nn.functional.relu(self.fc4(x))
        x = self.fc5(x)
        return x

net = Net(data_martix.shape[1])
net.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
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
for epoch in range(3000):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs = data['data'].float().to(device)
        labels = data['label'].long().to(device)

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
print("done")
plt.plot(losses[50:])
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
output = torch.argmax(net(torch.tensor(test[:,:,0]).float().to(device)), dim = 1).cpu().detach().numpy()
plt.title(str([f1_score(test[:,0,1],output),precision_score(test[:,0,1],output),recall_score(test[:,0,1],output)]))
plt.savefig("losses")

