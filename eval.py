import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from model import My_model
from tqdm import tqdm

# 独自のDataset
class MakeDataset(Dataset):
    def __init__(self, x_path, transform):
        self.x = np.load(x_path)
        self.transform = transform
    
    def __getitem__(self, index):
        x = torch.FloatTensor(self.x[index])
        x = torch.permute(x, (2, 0, 1))

        if self.transform:
            x = self.transform(x)
        
        return x
    
    def __len__(self):
        return len(self.x)
    
if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    os.chdir(os.path.dirname(__file__))

    x_test_path = "./data/camelyonpatch_level_2_split_test_x.npy"

    # 使用するデータの作成
    transform_test = transforms.Compose([
            transforms.CenterCrop(64),
            transforms.Resize(224),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    BATCH_SIZE = 16
    test_dataset = MakeDataset(x_path = x_test_path, transform = transform_test)
    test_loader = DataLoader(dataset = test_dataset, batch_size = BATCH_SIZE, num_workers = 2, shuffle = False)

    model = My_model()
    model.load_state_dict(torch.load('model_param.pth', weights_only = True))
    model = model.to(device)
    model.eval()

    pred_labels = []
    for inputs in tqdm(test_loader):
        inputs = inputs.to(device)
        outputs = model(inputs)
        outputs = outputs.flatten().cpu().detach().numpy()

        pred_labels.extend(outputs)
    
    pred_labels = np.array(pred_labels)

    np.savetxt('submit.csv', pred_labels, fmt = '%.6f')