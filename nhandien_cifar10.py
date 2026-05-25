import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
import torchvision.transforms as transforms
from cifar10 import CIFAR10Net

# Danh sách các nhãn trong CIFAR-10
classes = ('Máy bay', 'Ô tô', 'Chim', 'Mèo', 'Hươu', 'Chó', 'Ếch', 'Ngựa', 'Tàu thủy', 'Xe tải')

# Khởi tạo thiết bị và mô hình
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CIFAR10Net(num_classes=10).to(device)
model_path = 'cifar10_cnn_model.pth'

try:
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print("Da tai mo hinh CIFAR-10 thanh cong!")
except Exception as e:
    print(f"Loi khi tai mo hinh: {e}")
    print("Vui long dam bao ban da chay 'cifar10.py' de huan luyen mo hinh truoc.")
    exit()

# Transform cho ảnh CIFAR-10 (32x32, 3 kênh màu RGB, chuẩn hóa)
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class CIFAR10RecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Nhận diện ảnh CIFAR-10")
        
        # Tạo khung hiển thị ảnh
        self.canvas_width = 300
        self.canvas_height = 300
        self.canvas = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height, bg='lightgray')
        self.canvas.pack(pady=10)
        
        # Lưu trữ ảnh gốc
        self.image = None
        
        # Khung chứa nút bấm
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)
        
        self.btn_load = tk.Button(btn_frame, text="Tải ảnh lên", command=self.load_image, font=("Helvetica", 12))
        self.btn_load.grid(row=0, column=0, padx=10)
        
        self.btn_recognize = tk.Button(btn_frame, text="Nhận diện", command=self.recognize_image, font=("Helvetica", 12))
        self.btn_recognize.grid(row=0, column=1, padx=10)
        
        # Nhãn hiển thị kết quả
        self.label_result = tk.Label(root, text="Dự đoán: ...", font=("Helvetica", 16, "bold"), fg="blue")
        self.label_result.pack(pady=10)
        
    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.jfif *.webp *.gif *.tiff *.ico"), ("All files", "*.*")])
        if file_path:
            self.canvas.delete("all")
            # Mở ảnh bằng PIL
            img = Image.open(file_path).convert("RGB")
            self.image = img.copy() # Lưu lại để nhận diện
            
            # Thay đổi kích thước để hiển thị trên canvas cho đẹp (giữ tỷ lệ)
            img.thumbnail((self.canvas_width, self.canvas_height))
            self.tk_img = ImageTk.PhotoImage(img)
            
            # Căn giữa ảnh trên canvas
            x_center = (self.canvas_width - img.width) // 2
            y_center = (self.canvas_height - img.height) // 2
            self.canvas.create_image(x_center, y_center, anchor=tk.NW, image=self.tk_img)
            
            self.label_result.config(text="Dự đoán: ...")
            
    def recognize_image(self):
        if self.image is None:
            self.label_result.config(text="Vui lòng tải ảnh lên trước!")
            return
            
        # Biến đổi ảnh
        img_tensor = transform(self.image).unsqueeze(0).to(device)
        
        # Đưa vào mô hình dự đoán
        with torch.no_grad():
            output = model(img_tensor)
            # Lấy vị trí có giá trị lớn nhất (xác suất cao nhất) làm kết quả
            prediction_idx = torch.argmax(output, 1).item()
            predicted_class = classes[prediction_idx]
            
        self.label_result.config(text=f"Dự đoán: {predicted_class}")

if __name__ == "__main__":
    root = tk.Tk()
    app = CIFAR10RecognizerApp(root)
    root.mainloop()
