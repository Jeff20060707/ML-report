import numpy as np
import cv2
import os
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Input
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, precision_score
import seaborn as sns
import random


np.random.seed(7)


IMG_SIZE = (224, 224)
X = []
Z = []


flower = {
    "Daisy": r"D:\flower_photos/daisy",
    "Sunflower": r"D:\flower_photos/sunflowers",
    "Tulip": r"D:\flower_photos/tulips",
    "Dandelion": r"D:\flower_photos/dandelion",
    "Rose": r"D:\flower_photos/roses"
}

# 資料預處理：讀取圖片並調整大小
def make_train_data(flower_type, DIR):
    for filename in os.listdir(DIR):
        path = os.path.join(DIR, filename)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.resize(img, IMG_SIZE)
        X.append(img)
        Z.append(flower_type)

# 載入所有花卉圖片
for label, folder in flower.items():
    make_train_data(label, folder)

# 轉換 X 為 NumPy 陣列並正規化 (像素值介於 0~1)
X_array = np.array(X, dtype="float32")
X = X_array / 255.0

# 標籤轉換與 One-hot 編碼
encoder = LabelEncoder()
Z = encoder.fit_transform(Z)
Z = to_categorical(Z, num_classes=len(flower))


X_train, X_test, Z_train, Z_test = train_test_split(X, Z, test_size=0.3, random_state=42)

# 設定影像增強
data_gen = ImageDataGenerator(
    rotation_range=45,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_gen = data_gen.flow(X_train, Z_train, batch_size=64)


model = Sequential()
model.add(Input(shape=(224, 224, 3)))

model.add(Conv2D(32, kernel_size=(5, 5), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))    

model.add(Conv2D(128, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(flower), activation="softmax"))

# 編譯與訓練模型
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)

# 使用增強後的訓練數據進行訓練
history = model.fit(train_gen, epochs=25, validation_data=(X_test, Z_test), verbose=1, callbacks=[early_stopping, lr_scheduler])

# 評估模型
loss, accuracy = model.evaluate(X_train, Z_train, verbose=1)
print("訓練資料集的準確度 = {:.2f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, Z_test, verbose=1)
print("測試資料集的準確度 = {:.2f}".format(accuracy))

# 訓練與驗證損失曲線
loss = history.history["loss"]
epochs_range = range(1, len(loss)+1)
val_loss = history.history["val_loss"]
plt.figure()
plt.plot(epochs_range, loss, "bo-", label="Training Loss")
plt.plot(epochs_range, val_loss, "ro--", label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# 訓練與驗證準確度曲線
acc = history.history["accuracy"]
epochs_range = range(1, len(acc)+1)
val_acc = history.history["val_accuracy"]
plt.figure()
plt.plot(epochs_range, acc, "bo-", label="Training Accuracy")
plt.plot(epochs_range, val_acc, "ro--", label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# 混淆矩陣
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(Z_test, axis=1)

cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list(flower.keys()), yticklabels=list(flower.keys()))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# 精確率
precision_scores = precision_score(y_true, y_pred_classes, average=None)


# 隨機抽取測試樣本展示
num_samples = 5 
random_indices = random.sample(range(len(X_test)), num_samples)

plt.figure(figsize=(10, 5))
for i, idx in enumerate(random_indices):
    img = X_test[idx]
    true_label = list(flower.keys())[np.argmax(Z_test[idx])]
    pred_label = list(flower.keys())[np.argmax(model.predict(np.expand_dims(img, axis=0)))]

    plt.subplot(1, num_samples, i+1)
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"True: {true_label}\nPred: {pred_label}")

plt.suptitle("Random Test Sample Predictions")
plt.show()
