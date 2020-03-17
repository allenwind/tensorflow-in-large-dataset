import warnings
warnings.filterwarnings("ignore")
import glob
import collections
import itertools
import random
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import *

_THUContent = "/home/zhiwen/workspace/dataset/THUCTC/THUCNews/**/*.txt"
def load_THUCNews_content_label(file=_THUContent):    
    files = glob.glob(file)
    random.shuffle(files)

    # 获取所有标签
    labels = set()
    for file in files:
        label = file.rsplit("/", -2)[-2]
        labels.add(label)

    # label to id
    categoricals = {}
    for i, label in enumerate(labels):
        categoricals[label] = i

    def Xy_generator(files, with_label=True):
        for file in files:
            label = file.rsplit("/", -2)[-2]
            with open(file, encoding="utf-8") as fd:
                content = fd.read().strip()
            content = content.replace("\n", "").replace("\u3000", "")
            if with_label:
                yield content, categoricals[label]
            else:
                yield content

    return Xy_generator, files, categoricals

class CharTokenizer:

    def __init__(self, minfreq=16):
        self.char2id = {}
        self.MASK = 0
        self.UNKNOW = 1
        self.filters = set("!\"#$%&'()[]*+,-./，。！@·……（）【】")
        self.minfreq = minfreq

    def fit(self, X):
        # 建立词ID映射表
        chars = collections.defaultdict(int)
        for c in itertools.chain(*X):
            chars[c] += 1
        # 过滤低频词和特殊符号
        chars = {i:j for i, j in chars.items() \
                 if j >= self.minfreq and i not in self.filters}
        # 0 for MASK
        # 1 for UNK
        for i, c in enumerate(chars, start=2):
            self.char2id[c] = i

    def fit_generator(self, gen):
        self.fit(*[gen])

    def transform(self, X):
        # 转成ID序列
        ids = []
        for sentence in X:
            s = []
            for char in sentence:
                s.append(self.char2id.get(char, self.UNKNOW))
            ids.append(s)
        return ids

    def __len__(self):
        return self.vocab_size

    @property
    def vocab_size(self):
        return len(self.char2id) + 2

Xy_generator, files, classes = load_THUCNews_content_label()
num_classes = len(classes)
maxlen = 512

# 交叉验证
def split_index(size, scale=(0.8,0.1,0.1)):
    i = int(scale[0] * size)
    j = int((scale[0] + scale[1]) * size)
    return i, j

i, j = split_index(size=len(files))
files_train = files[:i]
files_val = files[i:j]
files_test = files[j:]

print("train tokenizer...")
tokenizer = CharTokenizer()
tokenizer.fit_generator(Xy_generator(files, with_label=False))

class DataGenerator:

    def __init__(self, files, repeat):
        self.files = files
        self.repeat = repeat

    def __call__(self):
        for _ in range(self.repeat):
            random.shuffle(self.files)
            for content, label in Xy_generator(self.files, with_label=True):
                content = content[:maxlen]
                content = tokenizer.transform([content])[0]
                label = tf.keras.utils.to_categorical(label, num_classes)
                yield content, label

num_words = len(tokenizer)
embedding_dims = 128
batch_size = 32
epochs = 20

def create_dataset(files, repeat=1):
    dataset = tf.data.Dataset.from_generator(
        generator=DataGenerator(files, repeat),
        output_types=(tf.int32, tf.int32)
    )

    dataset = dataset.padded_batch(
        batch_size=batch_size,
        padded_shapes=([maxlen], [None]),
        drop_remainder=True
    )
    return dataset

# 模型
inputs = Input(shape=(maxlen,))
x = Embedding(num_words, embedding_dims,
    embeddings_initializer="glorot_normal",
    input_length=maxlen)(inputs)
x = Dropout(0.1)(x)
x = Conv1D(filters=128,
           kernel_size=3,
           padding="same",
           activation="relu",
           strides=1)(x)
x = GlobalMaxPooling1D()(x)
x = Dense(128)(x)
x = Dropout(0.1)(x)
x = Activation("relu")(x)
outputs = Dense(num_classes, activation="softmax")(x)

model = Model(inputs, outputs)
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])
model.summary()

print("train model...")
dl_train = create_dataset(files_train, repeat=int(1e12))
dl_val = create_dataset(files_val)
model.fit(dl_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=dl_val,
          steps_per_epoch=len(files_train)//batch_size
)

print("evaluate model...")
dl_test = create_dataset(files_test)
model.evaluate(dl_test)
