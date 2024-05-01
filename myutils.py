import numpy as np

def train_test_split(X, y, ratio=0.2, random_seed=42):
    np.random.seed(random_seed)
    # 确定划分点
    n_samples = len(X)
    test_size = int(np.round(n_samples * ratio))
    # 随机打乱数据集的索引
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    # 划分训练集和测试集
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    return X_train, y_train, X_test, y_test

def transform(X , mean, std):
    return (X - mean) / std

class DataLoader:
    def __init__(self, X, y, batch_size ,shuffle=True):
        self.X = X
        if y is None:
            y = np.zeros((X.shape[0],))  # 如果 y 为 None，则自动用 0 填充
        self.y = y
        self.batch_size = batch_size
        self.num_samples = X.shape[0]
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))
        self.indices = np.arange(self.num_samples)
        self.shuffle = shuffle
        if self.shuffle:
            self.shuffle_data()

    def __len__(self):
        return self.num_samples

    def shuffle_data(self):
        np.random.shuffle(self.indices)

    def __iter__(self):
        self.current_batch = 0
        return self

    def __next__(self):
        if self.current_batch < self.num_batches:
            start_idx = self.current_batch * self.batch_size
            end_idx = min((self.current_batch + 1) * self.batch_size, self.num_samples)
            batch_indices = self.indices[start_idx:end_idx]
            batch_X = self.X[batch_indices]
            batch_y = self.y[batch_indices]
            self.current_batch += 1
            return batch_X, batch_y
        else:
            if self.shuffle:
                self.shuffle_data()
            self.current_batch = 0
            raise StopIteration


# batch_size = 7
# np.random.seed(42)
# data_loader = DataLoader(np.random.randn(19,2), None, batch_size)
#
# # Iterate through the data loader
# for batch_X, batch_y in data_loader:
#     # Use batch_X and batch_y for training or processing
#     print( batch_X[0] , batch_y[0])
#
#     print(batch_X.shape)
#     print(batch_y.shape)
#     pass
