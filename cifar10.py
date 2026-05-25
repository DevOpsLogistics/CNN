import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

# Cấu hình thiết bị (GPU hoặc CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Siêu tham số
num_epochs = 10
num_classes = 10
batch_size = 100
learning_rate = 0.001

# Chuẩn bị dữ liệu CIFAR-10
# CIFAR-10 là ảnh màu (3 kênh RGB), kích thước 32x32
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Chuẩn hóa cho 3 kênh màu
])

# Danh sách các nhãn trong CIFAR-10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Mạng CNN cho CIFAR-10
class CIFAR10Net(nn.Module):
    def __init__(self, num_classes=10):
        super(CIFAR10Net, self).__init__()
        # Input: 3 kênh (RGB) x 32 x 32
        
        self.conv_layers = nn.Sequential(
            # Lớp 1: 3 -> 32 kênh
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 32 x 16 x 16
            
            # Lớp 2: 32 -> 64 kênh
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 64 x 8 x 8
            
            # Lớp 3: 64 -> 128 kênh
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 128 x 4 x 4
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5), # Giảm overfitting
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1) # Flatten
        x = self.fc_layers(x)
        return x

model = CIFAR10Net(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

if __name__ == '__main__':
    model_path = 'cifar10_cnn_model.pth'
    
    if os.path.exists(model_path):
        print(f"Tìm thấy mô hình đã lưu tại '{model_path}'.")
        print("Đang tải mô hình...")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print("Tải mô hình thành công. Bỏ qua bước huấn luyện.")
    else:
        total_step = len(train_loader)
        print(f"Đang chạy trên thiết bị: {device}")
        print("Bắt đầu huấn luyện mô hình CIFAR-10...")

        for epoch in range(num_epochs):
            model.train() # Đặt mô hình ở chế độ huấn luyện
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)
                
                # Tiến
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Ngược và tối ưu
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if (i+1) % 100 == 0:
                    print(f'Epoch [{epoch+1}/{num_epochs}], Bước [{i+1}/{total_step}], Loss: {loss.item():.4f}')
                    
            # Lưu sau mỗi epoch nếu muốn (tùy chọn)
            epoch_model_filename = f'cifar10_cnn_model_epoch_{epoch+1}.pth'
            torch.save(model.state_dict(), epoch_model_filename)
            print(f"Đã lưu mô hình của epoch {epoch+1} vào '{epoch_model_filename}'")
            
        torch.save(model.state_dict(), model_path)
        print(f"Đã lưu trọng số mô hình cuối cùng vào '{model_path}'")

    # Đánh giá mô hình
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

        accuracy = 100 * correct / total
        print(f'Độ chính xác của mô hình trên 10000 ảnh test (CIFAR-10): {accuracy:.2f}%')
