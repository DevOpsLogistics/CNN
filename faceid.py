import os
import torch
from PIL import Image, ImageTk, ImageDraw
import tkinter as tk
from tkinter import filedialog, messagebox
import math

try:
    from facenet_pytorch import MTCNN, InceptionResnetV1
except ImportError:
    print("Vui lòng cài đặt facenet-pytorch trước: pip install facenet-pytorch")
    exit()

# Cấu hình thiết bị (GPU hoặc CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Khởi tạo MTCNN (để phát hiện khuôn mặt) và InceptionResnetV1 (để trích xuất đặc trưng khuôn mặt)
# keep_all=False nghĩa là chỉ lấy khuôn mặt rõ nhất trong ảnh nếu có nhiều khuôn mặt
mtcnn = MTCNN(keep_all=False, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

class FaceIDApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hệ thống Nhận diện Khuôn mặt (Face ID)")
        self.root.geometry("600x700")
        
        self.db_path = 'face_db.pt'
        self.face_db = {}
        
        # Tải cơ sở dữ liệu nếu đã có
        if os.path.exists(self.db_path):
            self.face_db = torch.load(self.db_path, map_location=device)
            print(f"Đã tải cơ sở dữ liệu với {len(self.face_db)} người.")
            
        # Khung giao diện
        self.canvas_width = 400
        self.canvas_height = 400
        self.canvas = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height, bg='lightgray')
        self.canvas.pack(pady=20)
        
        self.label_info = tk.Label(root, text=f"Dữ liệu hiện có: {len(self.face_db)} người", font=("Helvetica", 12))
        self.label_info.pack(pady=5)
        
        self.label_result = tk.Label(root, text="Chưa có dự đoán", font=("Helvetica", 16, "bold"), fg="blue")
        self.label_result.pack(pady=10)
        
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)
        
        self.btn_build_db = tk.Button(btn_frame, text="1. Chọn thư mục dữ liệu khuôn mặt", command=self.build_database, font=("Helvetica", 12))
        self.btn_build_db.grid(row=0, column=0, padx=10)
        
        self.btn_recognize = tk.Button(btn_frame, text="2. Chọn ảnh để nhận diện", command=self.recognize_face, font=("Helvetica", 12))
        self.btn_recognize.grid(row=0, column=1, padx=10)
        
    def get_embedding(self, img_path):
        """Trích xuất vector đặc trưng 512 chiều từ ảnh"""
        try:
            img = Image.open(img_path).convert('RGB')
            # Phát hiện và cắt khuôn mặt
            face = mtcnn(img)
            if face is not None:
                # Trích xuất đặc trưng
                emb = resnet(face.unsqueeze(0).to(device))
                return emb.detach()
        except Exception as e:
            print(f"Lỗi khi xử lý {img_path}: {e}")
        return None

    def build_database(self):
        """Xây dựng cơ sở dữ liệu từ một thư mục chứa ảnh (Tên file hoặc thư mục con là tên người)"""
        folder_path = filedialog.askdirectory(title="Chọn thư mục chứa ảnh khuôn mặt")
        if not folder_path:
            return
            
        self.label_info.config(text="Đang xử lý dữ liệu, vui lòng chờ...")
        self.root.update()
        
        new_db = {}
        valid_extensions = ('.png', '.jpg', '.jpeg', '.jfif', '.webp')
        
        # Duyệt qua các file trong thư mục
        for root_dir, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(valid_extensions):
                    img_path = os.path.join(root_dir, file)
                    
                    # Nếu ảnh nằm trong thư mục con, lấy tên thư mục con làm tên. Ngược lại lấy tên file.
                    if root_dir == folder_path:
                        name = os.path.splitext(file)[0]
                    else:
                        name = os.path.basename(root_dir)
                        
                    emb = self.get_embedding(img_path)
                    if emb is not None:
                        new_db[name] = emb
                        print(f"Đã thêm khuôn mặt: {name}")
                    else:
                        print(f"Không tìm thấy khuôn mặt rõ ràng trong: {file}")
                        
        if new_db:
            self.face_db.update(new_db)
            torch.save(self.face_db, self.db_path)
            messagebox.showinfo("Thành công", f"Đã cập nhật dữ liệu thành công! Tổng cộng: {len(self.face_db)} người.")
            self.label_info.config(text=f"Dữ liệu hiện có: {len(self.face_db)} người")
        else:
            messagebox.showwarning("Cảnh báo", "Không tìm thấy khuôn mặt nào hợp lệ trong thư mục.")
            self.label_info.config(text=f"Dữ liệu hiện có: {len(self.face_db)} người")

    def recognize_face(self):
        """Nhận diện khuôn mặt từ ảnh mới"""
        if not self.face_db:
            messagebox.showwarning("Cảnh báo", "Cơ sở dữ liệu trống. Vui lòng thêm dữ liệu khuôn mặt trước.")
            return
            
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.jfif *.webp")])
        if not file_path:
            return
            
        self.canvas.delete("all")
        
        # Hiển thị ảnh
        img = Image.open(file_path).convert('RGB')
        img_display = img.copy()
        img_display.thumbnail((self.canvas_width, self.canvas_height))
        self.tk_img = ImageTk.PhotoImage(img_display)
        
        x_center = (self.canvas_width - img_display.width) // 2
        y_center = (self.canvas_height - img_display.height) // 2
        self.canvas.create_image(x_center, y_center, anchor=tk.NW, image=self.tk_img)
        
        self.label_result.config(text="Đang phân tích...")
        self.root.update()
        
        # Lấy đặc trưng của ảnh đầu vào
        emb = self.get_embedding(file_path)
        if emb is None:
            self.label_result.config(text="Không tìm thấy khuôn mặt!")
            return
            
        # Tìm người giống nhất
        min_dist = float('inf')
        best_match = "Không xác định"
        
        for name, db_emb in self.face_db.items():
            # Tính khoảng cách Euclidean giữa 2 vector (càng nhỏ càng giống)
            dist = torch.dist(emb, db_emb).item()
            if dist < min_dist:
                min_dist = dist
                best_match = name
                
        # Ngưỡng (Threshold) để xác định xem có thực sự khớp không
        # Với vggface2, khoảng cách < 1.0 thường là cùng một người
        threshold = 1.0
        
        if min_dist <= threshold:
            self.label_result.config(text=f"Kết quả: {best_match}\n(Độ lệch: {min_dist:.2f})")
        else:
            self.label_result.config(text=f"Người lạ! (Giống {best_match} nhất nhưng độ lệch tới {min_dist:.2f})", fg="red")

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceIDApp(root)
    root.mainloop()
