import os
import shutil
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.family'] = 'Microsoft YaHei'
# 1. 自動切分資料集的函數
def split_dataset(base_dir, train_dir, val_dir, split_ratio=0.1):
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    for category in os.listdir(base_dir):
        category_path = os.path.join(base_dir, category)
        if os.path.isdir(category_path):
            images = os.listdir(category_path)
            random.shuffle(images)  # 隨機打亂圖像
            split_index = int(len(images) * (1 - split_ratio))  # 計算訓練集大小
            
            # 將圖像移動到訓練集和驗證集
            os.makedirs(os.path.join(train_dir, category), exist_ok=True)
            os.makedirs(os.path.join(val_dir, category), exist_ok=True)
            for img in images[:split_index]:
                shutil.copy(os.path.join(category_path, img), os.path.join(train_dir, category, img))
            for img in images[split_index:]:
                shutil.copy(os.path.join(category_path, img), os.path.join(val_dir, category, img))

# 2. 設定原始資料集路徑和切分後的路徑
base_directory = 'Animals/'  # 你的原始資料集路徑
train_directory = 'dataset/train_split/train'  # 訓練集路徑
val_directory = 'dataset/train_split/validation'  # 驗證集路徑

# 3. 切分資料集
split_dataset(base_directory, train_directory, val_directory)

# 4. 設定數據增強和預處理
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split = 0.2
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# 5. 載入訓練和驗證資料
train_generator = train_datagen.flow_from_directory(
    train_directory,
    target_size=(150, 150),  # 確保這裡的數據大小正確
    batch_size=32,  # 確保batch_size合適
    class_mode='categorical'  # 如果你的類別是多個，確保使用categorical
)

validation_generator = validation_datagen.flow_from_directory(
    val_directory,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# 6. 建立模型（根據需要進行調整）
model = keras.Sequential([
    layers.Input(shape=(150, 150, 3)),
    layers.Conv2D(32, (3, 3), activation='relu', padding = 'same',input_shape=(150, 150, 3)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding = 'same'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu', padding = 'same'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(3, activation='softmax')  # 根據你的類別數量進行調整
])

# 7. 編譯模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 8. 訓練模型
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=validation_generator,
)
loss, accuracy = model.evaluate(validation_generator)
print(f"val acc:{accuracy * 100:.2f}%")
# 9. 儲存模型
model.save('DogCatSnakemodel.keras')  # 使用Keras格式儲存


# 10. 預測圖片並顯示結果的函數
def predict_random_images(val_dir, model, num_samples):
    categories = os.listdir(val_dir)
    plt.figure(figsize=(12, 6))

    for i in range(num_samples):
        # 隨機選取一個類別
        category = random.choice(categories)
        category_path = os.path.join(val_dir, category)
        img_name = random.choice(os.listdir(category_path))
        img_path = os.path.join(category_path, img_name)
        
        # 預處理圖片
        img = image.load_img(img_path, target_size=(150, 150))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # 增加批次維度
        
        # 預測類別
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        predicted_category = train_generator.class_indices.keys()  # 預測的類別名稱

        # 每三個換一行
        picsperRow = 3
        rowsCounter = (num_samples + picsperRow - 1) // picsperRow

        # 顯示圖片和預測結果
        plt.subplot(rowsCounter, picsperRow, i + 1)
        plt.imshow(img)
        plt.title(f"預測: {list(predicted_category)[predicted_class]}\n實際: {category}")
        plt.axis('off')

    plt.show()

# 呼叫函數進行隨機預測
predict_random_images(val_directory, model, 9)


# 11. 繪製精確度和損失歷程圖
def plot_training_history(history):
    # 繪製準確度
    plt.figure(figsize=(12, 5))

    # 精確度圖
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='訓練精確度')
    plt.plot(history.history['val_accuracy'], label='驗證精確度')
    plt.title('模型精確度')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # 損失圖
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='訓練損失')
    plt.plot(history.history['val_loss'], label='驗證損失')
    plt.title('模型損失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# 呼叫函數繪製訓練歷程圖
plot_training_history(history)