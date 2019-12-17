import os

import numpy as np
import tensorflow as tf

# tensorflow 处理大数据集示例

num_epochs = 10
batch_size = 32
learning_rate = 0.001
data_dir = 'C:/datasets/cats_vs_dogs'
train_cats_dir = data_dir + '/train/cats/'
train_dogs_dir = data_dir + '/train/dogs/'
test_cats_dir = data_dir + '/valid/cats/'
test_dogs_dir = data_dir + '/valid/dogs/'

# wrapper


class DataGenerator:

    def gen(self):
        dog_files = os.listdir(self.train_dogs_dir)
        cat_files = os.listdir(self.train_cats_dir)
        files = np.array(dog_files + cat_files)
        labels = np.array([0] * len(dog_files) + [1] * len(cat_files))
        np.random.seed(109)
        np.random.shuffle(files)
        np.random.seed(109) # 保证标签对齐
        np.random.shuffle(labels)
        for file, label in zip(files, labels):
            yield (file, label)

def fn(filename, label):
    image_string = tf.io.read_file(filename)            # 读取原始文件
    image_decoded = tf.image.decode_jpeg(image_string)  # 解码JPEG图片
    image_resized = tf.image.resize(image_decoded, [256, 256]) / 255.0
    return image_resized, label

dataset = DataGenerator()
dataset = tf.data.Dataset.from_generator(dataset.gen()) \
                         .map(fn, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
                         .shuffle(buffer_size=256) \
                         .batch(batch_size) \
                         .prefetch(tf.data.experimental.AUTOTUNE)

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
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=[tf.keras.metrics.sparse_categorical_accuracy]
)

model.fit(train_dataset, epochs=num_epochs)

test_cat_filenames = tf.constant([test_cats_dir + filename for filename in os.listdir(test_cats_dir)])
test_dog_filenames = tf.constant([test_dogs_dir + filename for filename in os.listdir(test_dogs_dir)])
test_filenames = tf.concat([test_cat_filenames, test_dog_filenames], axis=-1)
test_labels = tf.concat([
    tf.zeros(test_cat_filenames.shape, dtype=tf.int32), 
    tf.ones(test_dog_filenames.shape, dtype=tf.int32)], 
    axis=-1)

test_dataset = tf.data.Dataset.from_tensor_slices((test_filenames, test_labels))
test_dataset = test_dataset.map(fn)
test_dataset = test_dataset.batch(batch_size)

print(model.metrics_names)
print(model.evaluate(test_dataset))