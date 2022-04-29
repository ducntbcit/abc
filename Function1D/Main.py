import torch
from CustomClass import CustomClass
from CustomDataSet import CustomDataSet
from sklearn.model_selection import train_test_split
from Train import train_model
import matplotlib.pyplot as plt


def target_function(x):
    return 2 ** x * torch.cos(2 ** (x))

if __name__=="__main__":
    HiddenNeurons=100
    Model = CustomClass(HiddenNeurons,1)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(Model.parameters(), lr=3e-4)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 1)
    lossMSE = torch.nn.MSELoss()
    lossL1 = torch.nn.L1Loss(reduction='mean')
    batch_size = 128
    X = torch.linspace(-5,5,5000)
    num_epochs = 2000
    #Y = np.zeros(X.shape) #target_function(X)
    Y = target_function(X)

    X.unsqueeze_(1)
    Y.unsqueeze_(1)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=60)
    test_DataLoader=torch.utils.data.DataLoader(CustomDataSet(x_test, y_test),batch_size=batch_size,shuffle=False)
    train_DataLoader = torch.utils.data.DataLoader(CustomDataSet(x_train, y_train), batch_size=batch_size, shuffle=False)

    train_model(Model,lossMSE,optimizer,scheduler,num_epochs,train_DataLoader,test_DataLoader,device)
    print("Loss L1: {}".format(lossL1(Model(X),Y)))

    X = torch.linspace(-5, 5, 5000)
    Y = target_function(X)
    x_flat=X.numpy()


    #plt.plot(x_flat, torch.flatten(Model(X.unsqueeze(1))).data.numpy(), '--', c='r', label="Prediction")
    plt.plot(x_flat,torch.flatten(Y).numpy(), c='g', label='Exact function')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(" 2 ** x * torch.cos(2 ** x) ")
    plt.legend()
    plt.show()

    plt.plot(x_flat, torch.flatten(Model(X.unsqueeze(1))).data.numpy(), '--', c='r', label="Function approximation")
    #plt.plot(x_flat,torch.flatten(Y).numpy(), c='g', label='Actual')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Function approximation by neural network ")
    plt.legend()
    plt.show()
    
    plt.plot(x_flat, torch.flatten(Model(X.unsqueeze(1))).data.numpy(), '--', c='r', label="Function approximation")
    plt.plot(x_flat,torch.flatten(Y).numpy(), c='g', label='Exact function')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(" Exact function and approximation function")
    plt.legend()
    plt.show()
