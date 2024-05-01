import os
import argparse
import numpy as np
import json
from utils.mnist_reader import load_mnist
from model import MLP_3layer_Model
from myutils import train_test_split , DataLoader , transform



'''
作业：从零开始构建三层神经网络分类器，实现图像分类

任务描述：
手工搭建三层神经网络分类器，在数据集[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)上进行训练以实现图像分类。

基本要求：
（1） 本次作业要求自主实现反向传播，不允许使用pytorch，tensorflow等现成的支持自动微分的深度学习框架，可以使用numpy；
（2） 最终提交的代码中应至少包含模型、训练、测试和参数查找四个部分，鼓励进行模块化设计；
（3） 其中模型部分应允许自定义隐藏层大小、激活函数类型，支持通过反向传播计算给定损失的梯度；
    训练部分应实现SGD优化器、学习率下降、交叉熵损失和L2正则化，并能根据验证集指标自动保存最优的模型权重；
    参数查找环节要求调节学习率、隐藏层大小、正则化强度等超参数，观察并记录模型在不同超参数下的性能；
    测试部分需支持导入训练好的模型，输出在测试集上的分类准确率（Accuracy）
'''




def train(model,train_loader,optim_args):
    model.train()
    loss_all = 0
    for batch_X, batch_y in train_loader:
        model.train()
        log_prob, loss = model(batch_X, batch_y, 'sum')
        model.backpropagation(optim_args=optim_args)
        loss_all += loss

    print('loss: ' , loss_all / len(train_loader) )
    return loss_all / len(train_loader)


def eval(model,eval_loader):
    model.eval()
    y_pred = None
    y_true = None
    for batch_X, batch_y in eval_loader:
        log_prob, _ = model(batch_X, batch_y, 'sum')
        if y_pred is None:
            y_pred = np.argmax(log_prob,axis=-1)
            y_true = batch_y.copy()
        else:
            y_pred = np.concatenate((y_pred, np.argmax(log_prob,axis=-1)), axis=0)
            y_true = np.concatenate((y_true, batch_y) , axis=0)
    return y_pred , y_true


if __name__ == '__main__':
    # 0. args init
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--lambda_w', type=float, default=1.0, help='l2-normalization weight')
    parser.add_argument('--data_dir', type=str, default='data/fashion', help='default dataset path')
    parser.add_argument('--batch_size', type=int, default=1024, help='')
    parser.add_argument('--save_dir', type=str, default='./results', help='default save path')
    parser.add_argument('--hidden_dim', type=int, default=256, help='中间层维度')
    parser.add_argument('--lr_decay_gamma', type=float, default=0.9, help='指数迭代参数')
    parser.add_argument('--MAX_ITER', type=int, default=99, help='最大迭代次数')
    args = parser.parse_args()

    from datetime import datetime
    formatted_date_time = datetime.now().strftime("d%m_%d_h%Hm%M")  # time string: month , day , hour , minute

    # 1. read data
    X_train, y_train = load_mnist(args.data_dir, kind='train') # [60000,28*28] , [60000,]
    X_test, y_test = load_mnist(args.data_dir, kind='t10k') # [10000, 28*28]

    X_train, y_train , X_valid, y_valid = train_test_split(X =X_train, y=y_train, ratio=0.2 )

    # X_train = X_train[0:10000]
    # y_train = y_train[0:10000]
    train_mean , train_std = np.mean(X_train) , np.std(X_train)
    #  用训练集的数据特征归一化
    X_train = transform(X_train, train_mean, train_std)
    X_valid = transform(X_valid, train_mean, train_std)
    X_test = transform(X_test, train_mean, train_std)
    eval_bz = 6000
    train_loader , valid_loader = DataLoader(X_train,y_train, args.batch_size,shuffle=True) , DataLoader(X_valid,y_valid,batch_size=eval_bz,shuffle=False)
    test_loader = DataLoader(X_test, y_test,batch_size=eval_bz,shuffle=False)


    X_test_transform = transform(X_test, np.mean(X_test) , np.std(X_test) )

    X , Y = X_test_transform[0:args.batch_size,:] , y_test[0:args.batch_size]
    # mean + valid +
    model = MLP_3layer_Model(input_dim=28*28, hidden_dim=128, cls_dim=10)

    y_pred , y_true = eval(model,eval_loader=valid_loader,)
    acc = (y_pred == y_true).sum() / len(y_pred)
    print('eval: ', acc)

    os.makedirs('./results' , exist_ok=True)
    filepath = './results/model_params_epoch{}.npz'
    # model.save(filepath)
    # new_model = MLP_3layer_Model(input_dim=28*28, hidden_dim=256, cls_dim=10)
    # new_model.load(filepath)
    # y_pred, y_true = eval(new_model, eval_loader=valid_loader, )
    # acc = (y_pred == y_true).sum() / len(y_pred)
    # print('new eval: ', acc)

    lr = 4e-3
    optim_args = {'lr':lr , 'lambda_w':0.01}
    # log_prob, loss = model(X_train, )
    # acc = (np.argmax(log_prob, axis=-1) == y_train ).sum() / len(y_train)
    # print(f'original acc: {acc}')

    # loss_func = CrossEntropy()
    # 划分 batch size
    import time
    start = time.time()
    restore_loss = []
    restore_valid_acc = []

    for epoch in range(args.MAX_ITER):
        loss_epoch = train(model,train_loader,optim_args)
        restore_loss.append(loss_epoch)
        print('*----------------*')
        # for idx, (batch_X, batch_y) in enumerate(train_loader):
        #     model.train(:)
        #     log_prob, loss = model(batch_X,batch_y)
        #     model.backpropagation(optim_args=optim_args)
        #
        #     end = time.time()
        #     print(idx,(end - start) )

        print(f'epoch {epoch}:')
        print('loss: ', loss_epoch )
        end = time.time()
        print( (end - start) / 60 )

        y_pred , y_true = eval(model, eval_loader=valid_loader)
        acc_valid = (y_pred == y_true).sum() / len(y_pred)
        restore_valid_acc.append(acc_valid)
        print('valid acc:' , acc_valid)

        with open('loss_acc.txt','a+',encoding='utf-8') as f:
            f.write(f'Epoch {epoch} , time { int( (end - start) / 60 )}  \n')
            f.write(str(loss_epoch))
            f.write('\n')
            f.write(str(acc_valid))
            f.write('\n')

        if epoch >= 60 and acc_valid < max(restore_valid_acc): # early stop
            print(epoch)
            break

        if len(restore_valid_acc) == 0 or acc_valid >= max(restore_valid_acc):
            model.save(filepath=filepath.format(epoch)) # 全存

        optim_args['lr'] = max( optim_args['lr'] * args.lr_decay_gamma, 5e-5)

    # restore loss
    # save loss / valid acc
    with open(f'loss_{formatted_date_time}.json', 'w', encoding='utf-8') as json_file:
        json.dump(restore_loss, json_file, ensure_ascii=False, indent=4)

    with open(f'acc.json_{formatted_date_time}', 'w', encoding='utf-8') as json_file:
        json.dump(restore_valid_acc, json_file, ensure_ascii=False, indent=4)

    print(restore_loss)
    print(restore_valid_acc)
