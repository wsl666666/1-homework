import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import fashion_mnist


class ThreeLayerNN:
    def __init__(self, input_size, hidden_size, output_size, activation='relu'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation

        # 初始化权重和偏置
        self.params = {}
        self.params['W1'] = np.random.randn(input_size, hidden_size) * 0.01
        self.params['b1'] = np.zeros((1, hidden_size))
        self.params['W2'] = np.random.randn(hidden_size, output_size) * 0.01
        self.params['b2'] = np.zeros((1, output_size))

    def forward(self, X):
        # 前向传播
        W1, b1, W2, b2 = self.params['W1'], self.params['b1'], self.params['W2'], self.params['b2']
        if self.activation == 'relu':
            self.z1 = np.dot(X, W1) + b1
            self.a1 = np.maximum(0, self.z1)  # ReLU激活函数
        elif self.activation == 'sigmoid':
            self.z1 = np.dot(X, W1) + b1
            self.a1 = 1 / (1 + np.exp(-self.z1))  # Sigmoid激活函数
        self.z2 = np.dot(self.a1, W2) + b2
        exp_scores = np.exp(self.z2)
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return self.probs

    def backward(self, X, y, learning_rate=0.01, reg_lambda=0.01):
        # 反向传播
        num_examples = X.shape[0]
        delta3 = self.probs
        delta3[range(num_examples), y] -= 1
        dW2 = np.dot(self.a1.T, delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        if self.activation == 'relu':
            delta2 = np.dot(delta3, self.params['W2'].T) * (self.a1 > 0)
        elif self.activation == 'sigmoid':
            delta2 = np.dot(delta3, self.params['W2'].T) * (self.a1 * (1 - self.a1))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        # 添加L2正则化项的梯度
        dW2 += reg_lambda * self.params['W2']
        dW1 += reg_lambda * self.params['W1']

        # 更新参数
        self.params['W1'] += -learning_rate * dW1
        self.params['b1'] += -learning_rate * db1
        self.params['W2'] += -learning_rate * dW2
        self.params['b2'] += -learning_rate * db2

    def calculate_accuracy(self, X, y):
        # 计算准确率
        num_examples = X.shape[0]
        W1, b1, W2, b2 = self.params['W1'], self.params['b1'], self.params['W2'], self.params['b2']
        z1 = np.dot(X, W1) + b1
        if self.activation == 'relu':
            a1 = np.maximum(0, z1)  # ReLU激活函数
        elif self.activation == 'sigmoid':
            a1 = 1 / (1 + np.exp(-z1))  # Sigmoid激活函数
        z2 = np.dot(a1, W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        accuracy = np.mean(np.argmax(probs, axis=1) == y)
        return accuracy


class ThreeLayerNNTrainer:
    def __init__(self, model):
        self.model = model
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.reg_lambda = None

    def train(self, X_train, y_train, X_val, y_val, num_epochs=100, batch_size=64, learning_rate=0.01, reg_lambda=0.01,
              lr_decay=0.95):
        self.reg_lambda = reg_lambda
        best_val_acc = 0
        best_params = None
        num_train = X_train.shape[0]
        lr = learning_rate
        val_accuracy = self.model.calculate_accuracy(X_val, y_val)

        for epoch in range(num_epochs):
            # 学习率下降
            lr *= lr_decay

            # 每个epoch开始前，打乱训练数据
            shuffle_index = np.random.permutation(num_train)
            X_train_shuffle = X_train[shuffle_index]
            y_train_shuffle = y_train[shuffle_index]

            for i in range(0, num_train, batch_size):
                # 获取一个小批量数据
                X_batch = X_train_shuffle[i:i + batch_size]
                y_batch = y_train_shuffle[i:i + batch_size]

                # 前向传播和反向传播
                probs = self.model.forward(X_batch)
                self.model.backward(X_batch, y_batch, lr, reg_lambda)

            # 计算训练集和验证集的损失
            train_loss = self.calculate_loss(X_train, y_train)
            val_loss = self.calculate_loss(X_val, y_val)

            # 计算验证集准确率
            val_accuracy = self.model.calculate_accuracy(X_val, y_val)

            # 保存训练过程中的损失和准确率
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)

            # 如果当前模型的验证集准确率优于之前的最佳模型，则保存当前模型参数
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                best_params = self.model.params

            print("Epoch {}: Training Loss = {:.4f}, Validation Loss = {:.4f}, Validation Accuracy = {:.2f}%".format(
                epoch + 1, train_loss, val_loss, val_accuracy * 100))

        self.model.params = best_params

    def calculate_loss(self, X, y):
        # 计算交叉熵损失
        num_examples = X.shape[0]
        W1, b1, W2, b2 = self.model.params['W1'], self.model.params['b1'], self.model.params['W2'], self.model.params['b2']
        z1 = np.dot(X, W1) + b1
        if self.model.activation == 'relu':
            a1 = np.maximum(0, z1)  # ReLU激活函数
        elif self.model.activation == 'sigmoid':
            a1 = 1 / (1 + np.exp(-z1))  # Sigmoid激活函数
        z2 = np.dot(a1, W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        corect_logprobs = -np.log(probs[range(num_examples), y])
        data_loss = np.sum(corect_logprobs)
        # 添加L2正则化项
        data_loss += 0.5 * self.reg_lambda * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
        return 1. / num_examples * data_loss

    def visualize_training_process(self):
        # 可视化训练过程中的损失和验证集上的准确率曲线
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.val_accuracies)
        plt.title('Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.suptitle('Training Process', fontsize=16)  # 添加总标题
        plt.show()


class ThreeLayerNNTester:
    def __init__(self, model):
        self.model = model

    def test(self, X_test, y_test):
        # 在测试集上进行测试
        test_accuracy = self.model.calculate_accuracy(X_test, y_test)
        print("Test Accuracy = {:.2f}%".format(test_accuracy * 100))


# 加载Fashion-MNIST数据集
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# 数据预处理
X_train = X_train.reshape(-1, 28 * 28) / 255.0
X_test = X_test.reshape(-1, 28 * 28) / 255.0

# 划分训练集、验证集和测试集
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# 创建并训练模型
model = ThreeLayerNN(input_size=784, hidden_size=128, output_size=10, activation='relu')
trainer = ThreeLayerNNTrainer(model)
trainer.train(X_train, y_train, X_val, y_val, num_epochs=20, batch_size=64, learning_rate=0.01, reg_lambda=0.01)

# 在测试集上进行测试
tester = ThreeLayerNNTester(model)
tester.test(X_test, y_test)

# 可视化训练过程
trainer.visualize_training_process()

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np


def visualize_weights(model):
    # 提取第一层的权重
    W1 = model.params['W1']

    # 获取滤波器数量和形状
    num_filters, filter_size = W1.shape[1], int(np.sqrt(W1.shape[0]))

    # 计算合适的 subplot 网格大小
    num_cols = min(8, num_filters)
    num_rows = (num_filters + num_cols - 1) // num_cols

    # 创建一个空白的画布
    plt.figure(figsize=(10, 10))

    # 循环遍历每个滤波器并显示出来
    for i in range(num_filters):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(W1[:, i].reshape(filter_size, filter_size), cmap='gray')
        plt.axis('off')
    plt.show()


# 创建模型并加载训练好的参数
model = ThreeLayerNN(input_size=784, hidden_size=128, output_size=10, activation='relu')
model.params = trainer.model.params  # 加载训练好的参数

# 可视化第一层的权重
visualize_weights(model)

import tensorflow as tf

# 假设您的模型对象名为 model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10)
])

# 编译模型
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, epochs=10)

# 保存模型权重
model.save_weights('model_weights.h5')
