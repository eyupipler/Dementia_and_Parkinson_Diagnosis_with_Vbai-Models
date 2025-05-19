import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._initialize_fc(num_classes)

    def _initialize_fc(self, num_classes):
        dummy_input = torch.zeros(1, 3, 448, 448)
        x = self.pool(self.relu(self.conv1(dummy_input)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        flattened_size = x.shape[1]
        self.fc1 = nn.Linear(flattened_size, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def load_model(device: str = 'cpu'):
    """
    Downloads and loads the pretrained SimpleCNN model for the 'c' version.
    """
    torch_device = torch.device(device)

    weights_path = hf_hub_download(
        repo_id="Neurazum/Vbai-DPA-2.2",
        filename="Vbai-DPA 2.2c.pt",
        repo_type="model"
    )

    model = SimpleCNN(num_classes=6).to(torch_device)
    state = torch.load(weights_path, map_location=torch_device)
    model.load_state_dict(state)
    model.eval()
    return model
