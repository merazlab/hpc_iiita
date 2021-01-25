import os
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms

def log(BASE_DIR , files=[], folders=[], temp_files=[], accuracy=00.00, exp_name="cls"):
    from datetime import datetime
    import shutil, os

    date_var = datetime.now().strftime("%m_%d-%I_%M%p")
    accuracy = str(accuracy)
    accuracy = accuracy.replace(".","_")
    exp_no = "a" + accuracy + "-" + date_var

    if not os.path.exists(os.path.join(BASE_DIR,'logs')):
        os.makedirs(os.path.join(BASE_DIR,'logs'))
    if not os.path.exists(os.path.join(BASE_DIR,'logs', exp_name)):
        os.makedirs(os.path.join(BASE_DIR,'logs', exp_name))
    if not os.path.exists(os.path.join(BASE_DIR,'logs', exp_name, exp_no)):
        os.makedirs(os.path.join(BASE_DIR,'logs', exp_name, exp_no))
    
    LOG_PATH = os.path.join(BASE_DIR,'logs', exp_name, exp_no)

    #Files copy paste in log
    for f in files:
        os.system('cp '+os.path.join(BASE_DIR, f)+' '+LOG_PATH)
    #Folder copy paste in logs
    for f in folders:
        # shutil.copytree(os.path.join(BASE_DIR, f), LOG_PATH)
        os.system('cp -r '+os.path.join(BASE_DIR, f)+' '+LOG_PATH)
    for f in temp_files:
        os.system('cp '+os.path.join(BASE_DIR, f)+' '+LOG_PATH)
        os.system('rm '+os.path.join(BASE_DIR, f))
        #out file
        # os.system('cp '+'/var/spool/mail/javed'+' '+LOG_PATH)


    print("------------------------------------------------------")
    print("--------------Logs_Created_Successfully----------------")
    print("-------------------------------------------------------")

class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()
    

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
io = IOStream(BASE_DIR + '/run.log')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 3
batch_size = 200
learning_rate = 0.001

# Image preprocessing modules
transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()])

# CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data/',
                                             train=True, 
                                             transform=transform,
                                             download=True)

test_dataset = torchvision.datasets.CIFAR10(root='./data/',
                                            train=False, 
                                            transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

   
model = Net().to(device)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# For updating learning rate
def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Train the model
total_step = len(train_loader)
curr_lr = learning_rate

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            prt = "Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}".format(epoch+1, num_epochs, i+1, total_step, loss.item())
            io.cprint(prt)
            print(prt)
    # Decay learning rate
    if (epoch+1) % 20 == 0:
        curr_lr /= 3
        update_lr(optimizer, curr_lr)

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    overall_accuracy = 100 * correct / total
    ptr = 'Accuracy of the model on the test images: {} %'.format(overall_accuracy)
    io.cprint(ptr)
    print(ptr)

# Save the model checkpoint
torch.save(model.state_dict(), 'resnet.ckpt')

####################################################################################
#making Log
#File list
files=['job.sh']

#Folder List
folders=[]

#Temp List
temp_files=['resnet.ckpt','run.log'] #remove from main after program finish


log(BASE_DIR, files, folders, temp_files, overall_accuracy)
