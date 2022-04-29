import torch

class CustomDataSet(torch.utils.data.Dataset):
    def __init__(self,input_array, output_array,transofrm=None):
        self.input_array=input_array
        self.output_array=output_array
    def __len__(self):
        return len(self.input_array)
    def __getitem__(self, index):
        return self.input_array[index], self.output_array[index]
