import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image

# 基本設定
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 資料前處理與載入 MNIST
def get_dataloader_for_norm(batch_size=64):
    transform_no_norm = transforms.Compose([
        transforms.ToTensor(), # [0,255] → [0,1] 並轉成 (C,H,W)
    ])


    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform_no_norm
    )

    train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=False)

    return train_loader

# 標準化函式
def compute_mean_std(loader):
    # 累積 sum 和 sum of squares
    n_pixels = 0
    channel_sum = 0.0
    channel_sum_sq = 0.0

    for images, _ in loader:
        # images shape: (batch, 1, 28, 28)
        # 把 batch、height、width 都展開在一起，只對 channel 維度做 mean
        # dim=(0,2,3) = 對 [batch, height, width] 取平均，保留 channel
        batch_pixels = images.size(0) * images.size(2) * images.size(3)

        channel_sum += images.sum(dim=[0, 2, 3])
        channel_sum_sq += (images ** 2).sum(dim=[0, 2, 3])
        n_pixels += batch_pixels

    # mean = E[x]
    mean = channel_sum / n_pixels
    # var = E[x^2] - (E[x])^2
    var = channel_sum_sq / n_pixels - mean ** 2
    std = torch.sqrt(var)

    print("Computed mean:", mean)
    print("Computed std:", std)

    return mean, std

# 重新進行標準化後的資料處理以及載入
def get_dataloaders(mean, std, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean.item(), std.item())
    ])

    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
        num_workers=1,      # 開啟 child process
        pin_memory=True # 固定記憶體位置
    )

    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform,
        num_workers=1,      # 開啟 child process
        pin_memory=True # 固定記憶體位置
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
     
    return train_loader, test_loader

# 定義 CNN 模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # input: (1, 28, 28)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        # output: (32, 28, 28) → MaxPool2d(kernel_size=2) → (32, 14, 14)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # output: (64, 14, 14) → MaxPool2d(kernel_size=2) → (64, 7, 7)

        self.dropout = nn.Dropout(0.5)

        # 全連接層：64 * 7 * 7 = 3136
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 類（數字 0~9）

    def forward(self, x):
        # 第一層卷積 + ReLU + Pooling
        x = self.conv1(x)           # (batch, 1, 28, 28) → (batch, 32, 28, 28)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)      # → (batch, 32, 14, 14)

        # 第二層卷積 + ReLU + Pooling
        x = self.conv2(x)           # → (batch, 64, 14, 14)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)      # → (batch, 64, 7, 7)

        # 展平（Flatten）
        x = x.view(x.size(0), -1)   # (batch, 64*7*7 = 3136)

        x = self.dropout(x)
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)             # (batch, 10) logits
        return x

# 訓練 & 評估函數
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        # 歸零梯度
        optimizer.zero_grad()

        # Forward
        outputs = model(images)          # (batch, 10)
        loss = criterion(outputs, labels)

        # Backward
        loss.backward()
        optimizer.step()

        # 累計 loss 與 accuracy
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # 減少記憶體與加速
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc




def predict_image(model, image_path, device, mean, std, invert=False):
    # 實際測試手寫圖片
    inference_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),      # 轉成 1 channel 灰階
        transforms.Resize((28, 28)),                      # 縮放到 28x28
        transforms.ToTensor(),                            # [0,255] → [0,1]
        transforms.Normalize(mean.item(), std.item())        # 跟訓練時一樣
    ])

    model.eval()  # 推論模式（關掉 Dropout 等）
    img = Image.open(image_path)
    img = inference_transform(img)   # shape: (1, 28, 28)

    if invert:     # 如果圖片是「黑色字 + 白色背景」，需反向
        img = 1.0 - img

    img = img.unsqueeze(0).to(device) # 增加 batch 維度：變成 (1, 1, 28, 28)

    with torch.no_grad():
        outputs = model(img)                 # shape: (1, 10)，logits
        probs = F.softmax(outputs, dim=1)    # 轉成機率
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_class].item()

    return pred_class, confidence

def main():
    batch_size = 64
    num_epochs = 8
    learning_rate = 1e-3
    momentum = 0.9
    weight_decay = 5e-4
    
    device = get_device()

    train_loader = get_dataloader_for_norm(batch_size)
    mean, std = compute_mean_std(train_loader)
    train_loader, test_loader =  get_dataloaders(mean, std, batch_size)
    
    model = SimpleCNN().to(device)

    # 損失函數與優化器
    criterion = nn.CrossEntropyLoss()             # 適用於 multi-class 分類
    optimizer = optim.Adam(
        model.parameters(), 
        momentim = momentum,
        weight_decay = weight_decay,
        lr=learning_rate)
    
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        print(f"Epoch [{epoch}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% "
            f"|| Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}%")
        
    root = "./data/test_number"
    paths = [f"{root}/0.jpg", f"{root}/3.jpg", f"{root}/7.jpg"]
    for p in paths:
        pred, conf = predict_image(model, p, device, invert=True)
        print(p, "→", pred, f"({conf*100:.2f}%)") 
    
if __name__ == "__main__":
    main()