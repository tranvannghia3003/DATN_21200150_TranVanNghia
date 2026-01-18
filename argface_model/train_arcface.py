import os
from argface_model.argface_classifier import ArcFaceClassifier

def main():
    # --- 1. Thiết lập các đường dẫn cần thiết ---
    root = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(root, 'build')
    
    arcface_dataset = os.path.join(build_dir, 'arcface_train_dataset')
    arcface_model_dir = os.path.join(build_dir, 'arcface_train') # Thư mục riêng cho ArcFace train
    arcface_model_path = os.path.join(arcface_model_dir, 'arcface_model.pth')
    curves_save_path = os.path.join(arcface_model_dir, 'arcface_training_curves.png')

    # Tạo thư mục lưu trữ nếu chưa có
    os.makedirs(arcface_model_dir, exist_ok=True)

    # Kiểm tra dataset
    if not os.path.exists(arcface_dataset):
        print(f"Lỗi: Không tìm thấy thư mục dataset tại: {arcface_dataset}")
        return

    # --- 2. Khởi tạo và Huấn luyện ArcFace ---
    print("Khởi tạo ArcFace Classifier...")
    classifier = ArcFaceClassifier(arcface_dataset, arcface_model_dir, arcface_model_path)
    
    # Kiểm tra xem có model cũ không để tiếp tục train hoặc train mới
    if classifier.model_exists():
        print("Tìm thấy model ArcFace cũ. Sẽ tiếp tục huấn luyện (fine-tuning).")
        classifier.load_model()
    else:
        print("Không tìm thấy model ArcFace. Sẽ huấn luyện từ đầu.")
        classifier.initialize_model()

    # Trích xuất đặc trưng trước khi train
    classifier.extract_features()
    
    # Bắt đầu huấn luyện
    print("Bắt đầu quá trình huấn luyện ArcFace...")
    classifier.train(num_epochs=50, lr=0.001) # Bạn có thể thay đổi hyperparameters ở đây
    print("Hoàn thành huấn luyện ArcFace.")

    # --- 3. Vẽ và lưu biểu đồ huấn luyện ---
    print("Đang vẽ và lưu biểu đồ huấn luyện...")
    classifier.plot_training_metrics(save_path=curves_save_path)
    print(f"Đã lưu biểu đồ vào: {curves_save_path}")

if __name__ == '__main__':
    main()