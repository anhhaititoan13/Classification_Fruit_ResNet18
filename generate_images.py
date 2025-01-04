import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import numpy as np

# Đường dẫn đến thư mục chứa các loại động vật
base_dir = "datasets/train"

# Danh sách các loại động vật
animal_types = ["meo"]

# Số lượng ảnh mới muốn tạo ra từ mỗi ảnh gốc
augmentation_factor = 10

# Khởi tạo ImageDataGenerator với các tham số tương tự như trước
aug = ImageDataGenerator(
	rotation_range=30,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# Lặp qua từng loại động vật
for animal_type in animal_types:
    print(f"[INFO] Generating images for {animal_type}...")
    
    # Tạo đường dẫn đầy đủ đến thư mục chứa ảnh của loại động vật hiện tại
    animal_dir = os.path.join(base_dir, animal_type)
    
    # Đảm bảo rằng thư mục tồn tại
    if not os.path.exists(animal_dir):
        print(f"[ERROR] Directory {animal_dir} not found.")
        continue
    
    # Lặp qua mỗi file ảnh trong thư mục của loại động vật hiện tại
    for root, dirs, files in os.walk(animal_dir):
        for file in files:
            # Load ảnh và chuyển đổi thành mảng NumPy
            image_path = os.path.join(root, file)
            image = load_img(image_path)
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            
            # Tạo thư mục để lưu ảnh được tạo ra từ ảnh gốc
            save_dir = os.path.join(root, "generated")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            # Tạo generator dữ liệu từ ảnh hiện tại
            imageGen = aug.flow(image, batch_size=1, save_to_dir=save_dir, save_prefix=file.split('.')[0], save_format="jpg")
            
            # Tạo thêm ảnh đến khi đạt đến augmentation_factor
            count = 0
            for _ in imageGen:
                count += 1
                if count == augmentation_factor:
                    break
