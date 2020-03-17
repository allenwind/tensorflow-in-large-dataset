import warnings
warnings.filterwarnings("ignore")
import os
import glob
import numpy as np
import tensorflow as tf

prefix = "/home/zhiwen/workspace/dataset/fastai-datasets-cats-vs-dogs/"
train_dogs_dir = prefix + "train/dogs/*.jpg"
train_cats_dir = prefix + "train/cats/*.jpg"
valid_dogs_dir = prefix + "valid/dogs/*.jpg"
valid_cats_dir = prefix + "valid/cats/*.jpg"


def load_files(dogs_dir, cats_dir):
    dog_files = glob.glob(dogs_dir) # or use tf.data.Dataset.list_files
    cat_files = glob.glob(cats_dir)
    files = np.array(dog_files + cat_files)
    labels = np.array([0] * len(dog_files) + [1] * len(cat_files))
    np.random.seed(109)
    np.random.shuffle(files)
    np.random.seed(109) # 保证标签对齐
    np.random.shuffle(labels)
    labels = tf.keras.utils.to_categorical(labels)
    return files, labels

def fn(filename, label):
    image_string = tf.io.read_file(filename)            # 读取原始文件
    image_decoded = tf.image.decode_jpeg(image_string)  # 解码JPEG图片
    image_resized = tf.image.resize(image_decoded, [256, 256]) / 255.0
    return image_resized, label

batch_size = 32
# 训练集
files, labels = load_files(train_dogs_dir, train_cats_dir)
train_dataset = tf.data.Dataset.from_tensor_slices((files, labels)) \
                         .map(fn, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
                         .shuffle(buffer_size=256) \
                         .batch(batch_size) \
                         .prefetch(tf.data.experimental.AUTOTUNE)

# 验证集
files, labels = load_files(valid_dogs_dir, valid_cats_dir)
valid_dataset = tf.data.Dataset.from_tensor_slices((files, labels)) \
                               .map(fn, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
                               .batch(batch_size) \

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(256, 256, 3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 5, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["acc"]
)

epochs = 10
model.fit(train_dataset, epochs=epochs, validation_data=valid_dataset)
