import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageDraw, ImageOps, ImageTk
import torch
import torchvision.transforms as transforms
from tichchap import ConvNet

# Khởi tạo thiết bị và mô hình
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ConvNet(num_classes=10).to(device)
model_path = 'mnist_cnn_model.pth'

try:
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print("Da tai mo hinh thanh cong!")
except Exception as e:
    print(f"Loi khi tai mo hinh: {e}")
    print("Vui long dam bao ban da chay 'tichchap.py' de huan luyen mo hinh truoc.")
    exit()

# Transform cho ảnh giống như MNIST (28x28, 1 kênh màu grayscale, chuẩn hóa)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Nhận diện chữ số viết tay")
        
        # Tạo khung vẽ (Canvas) 280x280 cho dễ vẽ
        self.canvas_width = 280
        self.canvas_height = 280
        self.canvas = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height, bg='white', cursor='cross')
        self.canvas.pack(pady=10)
        
        # Gắn sự kiện kéo chuột để vẽ
        self.canvas.bind("<B1-Motion>", self.draw_lines)
        
        # Ảnh ẩn để lưu hình vẽ và đưa vào mô hình
        self.image = Image.new("RGB", (self.canvas_width, self.canvas_height), (255, 255, 255))
        self.draw = ImageDraw.Draw(self.image)
        
        # Khung chứa các nút bấm
        btn_frame = tk.Frame(root)
        btn_frame.pack()
        
        self.btn_recognize = tk.Button(btn_frame, text="Nhận diện", command=self.recognize_digit, font=("Helvetica", 12))
        self.btn_recognize.grid(row=0, column=0, padx=5)
        
        self.btn_clear = tk.Button(btn_frame, text="Xóa (Vẽ lại)", command=self.clear_canvas, font=("Helvetica", 12))
        self.btn_clear.grid(row=0, column=1, padx=5)
        
        self.btn_load = tk.Button(btn_frame, text="Tải ảnh có sẵn", command=self.load_image, font=("Helvetica", 12))
        self.btn_load.grid(row=0, column=2, padx=5)
        
        # Nhãn hiển thị kết quả
        self.label_result = tk.Label(root, text="Dự đoán: ", font=("Helvetica", 20, "bold"), fg="blue")
        self.label_result.pack(pady=20)
        
    def draw_lines(self, event):
        x, y = event.x, event.y
        # Bán kính bút vẽ
        r = 12 
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill='black', outline='black')
        self.draw.ellipse([x-r, y-r, x+r, y+r], fill='black')
        
    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, self.canvas_width, self.canvas_height], fill="white")
        self.label_result.config(text="Dự đoán: ")
        
    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.jfif *.webp *.gif *.tiff *.ico"), ("All files", "*.*")])
        if file_path:
            # Xóa canvas cũ
            self.clear_canvas()
            # Mở ảnh và thay đổi kích thước cho vừa khung
            img = Image.open(file_path).resize((self.canvas_width, self.canvas_height))
            self.image = img.convert("RGB")
            self.draw = ImageDraw.Draw(self.image)
            
            # Hiển thị lên canvas
            self.tk_img = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)
            
    def recognize_digit(self):
        # Ảnh huấn luyện MNIST có nền đen, chữ trắng. Nên ta cần nghịch đảo màu từ ảnh vẽ (nền trắng chữ đen).
        img_inverted = ImageOps.invert(self.image.convert('L'))
        
        # Để mô hình nhận diện tốt hơn, ta cần căn giữa và phóng to nét vẽ (giống chuẩn MNIST)
        bbox = img_inverted.getbbox() # Lấy khung bao quanh nét vẽ
        if bbox:
            img_cropped = img_inverted.crop(bbox)
            
            # Thay đổi kích thước sao cho chiều dài nhất bằng 20 pixel (chuẩn MNIST là 20x20 nằm trong khung 28x28)
            ratio = 20.0 / max(img_cropped.size)
            new_size = (int(img_cropped.size[0] * ratio), int(img_cropped.size[1] * ratio))
            
            # Dùng LANCZOS hoặc BICUBIC để resize không bị vỡ hạt quá nhiều
            from PIL import Image
            img_resized = img_cropped.resize(new_size, Image.Resampling.LANCZOS)
            
            # Tạo ảnh mới 28x28 nền đen (0)
            final_img = Image.new('L', (28, 28), 0)
            
            # Tính toán vị trí để dán ảnh đã resize vào chính giữa khung 28x28
            paste_x = (28 - new_size[0]) // 2
            paste_y = (28 - new_size[1]) // 2
            final_img.paste(img_resized, (paste_x, paste_y))
            
            # Thay vì dùng lại transform cũ (có resize), ta chỉ chuyển ToTensor và Normalize
            custom_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            img_tensor = custom_transform(final_img).unsqueeze(0).to(device)
        else:
            # Nếu chưa vẽ gì
            img_tensor = transform(img_inverted).unsqueeze(0).to(device)
        
        # Đưa vào mô hình dự đoán
        with torch.no_grad():
            output = model(img_tensor)
            # Lấy vị trí có giá trị lớn nhất (xác suất cao nhất) làm kết quả
            prediction = torch.argmax(output, 1).item()
            
        self.label_result.config(text=f"Dự đoán: {prediction}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()
