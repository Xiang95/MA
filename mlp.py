# MLP
import numpy as np
import torch as pt
import torchvision as ptv
import torch.nn.functional as F
import torch.nn as nn

train_set = ptv.datasets.MNIST("../../pytorch_database/mnist/train",train=True,transform=ptv.transforms.ToTensor(),download=True)
test_set = ptv.datasets.MNIST("../../pytorch_database/mnist/test",train=False,transform=ptv.transforms.ToTensor(),download=True)

train_dataset = pt.utils.data.DataLoader(train_set,batch_size=100)
test_dataset = pt.utils.data.DataLoader(test_set,batch_size=100)

class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size,output_size):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden = torch.nn.Linear(self.inputsize, self.hidden_size)
        self.hidden2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.out = torch.nn.Linear(self.hidden_size, output_size)

    def forward(self, din):
        din = din.view(-1, self.input_size)
        dout = F.relu(self.hidden(din))
        dout = F.relu(self.hidden2(dout))
        # how to choose dim???
        dout = F.softmax(self.out(dout), dim=1)
        return dout

model = MLP(28*28, )
# define training
# loss func and optim
optimizier = pt.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
lossfunc = nn.CrossEntropyLoss().cuda()

# accuarcy
def AccuarcyCompute(pred, label):
    pred = pred.cpu().data.numpy()
    label = label.cpu().data.numpy()
#   printC(pred.shape(),label.shape())
    test_np = (np.argmax(pred, 1)==label)
    test_np = np.float32(test_np)
    return np.mean(test_np)

for x in range(4):
    for i, data in enumerate(train_dataset):
        optimizier.zero_grad()
        (inputs, labels) = data
        inputs = pt.autograd.Variable(inputs).cuda()
        labels = pt.autograd.Variable(labels).cuda()

        outputs = model(inputs)

        loss = lossfunc(outputs, labels)
        loss.backward()

        optimizier.step()

        if i % 100 ==0:
            print(i, ':', AccuarcyCompute(outputs, labels))

accuarcy_list = []
for i,(inputs,labels) in enumerate(test_dataset):
    inputs = pt.autograd.Variable(inputs).cuda()
    labels = pt.autograd.Variable(labels).cuda()
    outputs = model(inputs)
    accuarcy_list.append(AccuarcyCompute(outputs,labels))
print(sum(accuarcy_list) / len(accuarcy_list))