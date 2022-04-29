import torch
from CustomClass import CustomClass


HiddenNeurons=50
NumberOfInputs=2
Model = CustomClass(NumberOfInputs, HiddenNeurons,1)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(Model.parameters(), lr=3e-4)
scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 1)
lossMSE = torch.nn.MSELoss(reduction='mean')
lossL1 = torch.nn.L1Loss(reduction='mean')
batch_size = 128
NumberOfX=1000
NumberOfY=1000
num_epochs = 70