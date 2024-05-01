import numpy as np

# *-------------- 模块（积木 * 包括 loss_func） ---------------------*

class Linear:
    def __init__(self , in_dim , out_dim , bias=True):
        self.status = 'train' # 'eval'
        self.require_grad = True
        self.grad_saved_x = None

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.need_bias = bias

        self.W = None # TODO Kaming
        self.b = None
        self.KaimingInit()
        pass

    def __call__(self, input,):
        return self.forward(input)

    def clear_grad(self):
        self.grad_saved_x = None

    def KaimingInit(self):
        '''以全零初始化b , 以随机正交矩阵初始化W , relu 会将模长再缩小 1/sqrt(2) ;
        苏神文章：https://spaces.ac.cn/archives/7180 解释 恺明大佬的初始化
         `当 m≥n 时，从任意的均值为0、方差为 1/m 的分布 p(x) 中独立重复采样出来的 m×n 矩阵，近似满足W^TW=I`
        简单起见，我们一直假设 m > n , Linear 主打一个降维操作
        '''

        if self.need_bias:
            self.b = np.zeros( self.out_dim )
        # 因此，选择一个 unif(limit , -limit) , 使得 2/in_dim = var = a^2 / 3 --> a = \sqrt( 6 / in_dim )
        limit = np.sqrt(6/self.in_dim)
        # 使用均匀分布初始化权重
        self.W = np.random.uniform(-limit, limit, (self.in_dim , self.out_dim))
        print('init ...')
        return


    def forward(self , input ):
        '''
          W^T x + b \in R^{out_dim} , W \in R^{in_dim , out_dim}
        :param input: [bz , in_dim]
        :return:
        '''
        out = input @ self.W
        if self.need_bias:
            out += self.b
        if self.require_grad and self.status:
            self.grad_saved_x = input
        return out


    def backpropagation(self, back_grad , optim_args):
        '''
        :param back_grad: (\partial loss / \partial f1) * (\partial f1 / \partial f2) ...  [bz , out_dim]
        :param lr:
        :return:
        '''
        # update params (SGD), then propagation...

        # f(x) \in R^m , x \in R^n 用 Jacobi Matrix \in R^{m*n} 计算 ,  梯度相当于 Jacobi 的转置 .
        # x \in R^{in_dim} , W \in R^{in_dim, out_dim} , b \in R^{out_dim}
        # 记 f_x(W , b) = W^Tx + b \in R^{out_dim}
        #   \partial f / \partial x  = W^T ,
        #   将 W 看成向量，与 Delta W 作内积
        #   \partial f / \partial w  =  { [0,x(第i位),0..0] \in R{in_dim , out_dim} }_i=1..out_dim  , \in [ out_dim , in_dim , out_dim ] ,
        #   \partial f / \partial b  = I_{out_dim}

        # TODO
        lr , lambda_w = optim_args['lr'] , optim_args['lambda_w']
        bz = back_grad.shape[0]
        new_back_grad = back_grad @ self.W.T # [bz , in_dim] *对 X 求导
        # update weights

        # # Failure Try 1: memory occipied
        # partial_intermediate_w = np.zeros((bz,self.out_dim,self.out_dim,self.in_dim))  # [bz, out_dim , in_dim , out_dim ]
        # # 将向量x赋值给对角线
        # for bz_idx in range(bz):
        #     for i in range(self.out_dim):
        #         partial_intermediate_w[bz_idx, i , i, :] = self.grad_saved_x[bz_idx]
        # w_grad_0 = np.einsum( "ij,ijmn->imn", back_grad, partial_intermediate_w ).transpose() # TODO check

        w_grad = np.zeros((bz,self.in_dim,self.out_dim))
        for bz_idx in range(bz):
            # 优化内存占用
            x_tmp = self.grad_saved_x[bz_idx]
            for i in range(self.out_dim):
                w_grad[bz_idx,:, i] = back_grad[bz_idx,i] * x_tmp
        w_grad = w_grad.transpose((1,2,0))

        # 随机梯度下降 , 对 batch 取平均
        self.W = self.W - lr* (w_grad.mean(axis=-1) + 2*lambda_w * self.W )
        if self.need_bias:
            grad_b = ( back_grad @ np.ones((self.out_dim , self.out_dim)) ).transpose()
            self.b = self.b - lr * grad_b.mean(axis=-1)
        return new_back_grad # [bz , c] when training


class Relu:
    def __init__(self):
        self.status = 'train' # 'eval'
        self.require_grad = True
        self.grad_saved = None
        pass

    def __call__(self, x):
        return self.forward(x)

    def forward(self, input):
        '''
        :param input: [bz , dim]
        :return:
        '''
        if self.require_grad:
            self.grad_saved = (input > 0).astype(float)
        return np.maximum(input, 0)

    def backpropagation(self , back_grad , optim_args=None):
        bz , dim = self.grad_saved.shape
        relu_grad = np.zeros( (bz , dim, dim))
        for bz_idx in range(bz):
            relu_grad[bz_idx] = np.diag( self.grad_saved[bz_idx] ) # [bz , dim, dim ]
        new_back_grad = np.einsum("bn,bnk->bk" , back_grad, relu_grad )
        return new_back_grad # Jacobi

    def clear_grad(self):
        self.grad_saved = None

class CrossEntropy:
    def __init__(self):
        self.status = 'train' # 'eval'
        self.require_grad = True
        self.grad_saved = None
        pass

    def __call__(self, input, Y, loss_type='mean' ):
        return self.forward(input,Y,loss_type=loss_type)

    def clear_grad(self):
        self.grad_saved = None

    def convertOneHot(self,Y , cls_dim):
        '''
        :param Y: [bz,]
        :return: Y_oneHot: [bz , cls_dim ]
        '''
        Y_oneHot = np.eye(cls_dim)[Y]
        return Y_oneHot

    def forward(self , input, Y , loss_type='mean'):
        '''
        :param X: [bz , cls_dim]
        :param Y: [bz]
        :param loss_type: 'mean' , 'sum'
        :return:
        '''
        assert loss_type in ('mean' , 'sum')
        bz , cls_dim = input.shape
        exp_X = np.exp(input - np.max(input, axis=-1, keepdims=True))
        P = exp_X / np.sum(exp_X, axis=-1, keepdims=True) # Probability's. [bz , cls_dim]
        Y_oneHot = self.convertOneHot(Y , cls_dim=cls_dim)
        log_P = np.log(P)
        loss_sum = -np.diag( Y_oneHot @ log_P.T ).sum()
        loss_mean = loss_sum / bz

        if self.require_grad and self.status:
            # 存 / 算梯度
            self.grad_saved = -(Y_oneHot - P) # [bz , cls_dim]
        return loss_mean if loss_type == 'mean' else loss_sum


    def backpropagation(self, optim_args=None):
        return self.grad_saved # [bz , cls_dim] when training


# *-------------- 最终模型 ---------------------*
class MLP_3layer_Model:
    '''
        为了方便反向传播，直接把 loss 也放进来了
    '''
    def __init__(self,input_dim , hidden_dim, cls_dim, use_bias = True):
        self.linear_1 = Linear(input_dim , hidden_dim , bias=use_bias)
        self.relu_1 = Relu()
        self.linear_2 = Linear(hidden_dim , hidden_dim , bias=use_bias)
        self.relu_2 = Relu()
        self.linear_3 = Linear(hidden_dim, cls_dim, bias=use_bias)
        self.loss_func = CrossEntropy()
        self.modules_list = [self.linear_1,self.relu_1,self.linear_2,self.relu_2,self.linear_3,self.loss_func]
        pass

    def __call__(self, inputs , labels=None, loss_type='sum'):
        return self.forward(inputs,labels,loss_type)


    def forward(self, inputs , labels=None, loss_type='sum'):
        '''
        :param inputs: [bz , in_dim]
        :param labels: [bz]
        :return:
        '''
        out = self.relu_1( self.linear_1( inputs ) )
        out = self.relu_2( self.linear_2( out) )
        out = self.linear_3(out)
        if labels is None: # without loss
            return out , None
        loss = self.loss_func(out , labels , loss_type=loss_type)
        return out, loss

    def backpropagation(self , optim_args):
        # backpropagation and update weights
        back_grad = self.loss_func.backpropagation() # [bz , cls_dim ]
        back_grad = self.linear_3.backpropagation(back_grad = back_grad, optim_args=optim_args) # []
        back_grad = self.relu_2.backpropagation(back_grad) #
        back_grad = self.linear_2.backpropagation(back_grad,optim_args=optim_args) #
        back_grad = self.relu_1.backpropagation(back_grad) #
        back_grad = self.linear_1.backpropagation(back_grad,optim_args=optim_args)
        # Or to be more efficient .
        # self.modules_list

        return

    def eval(self):
        for module in self.modules_list:
            module.status = 'eval'
        return

    def train(self):
        for module in self.modules_list:
            module.status = 'train'
        return

    def save(self,filepath):
        parameters = {}
        for i, module in enumerate(self.modules_list):
            if hasattr(module, 'W'):
                parameters[f'W{i}'] = module.W
            if hasattr(module, 'b') and module.b is not None:
                parameters[f'b{i}'] = module.b
        np.savez(filepath, **parameters)
        return


    def load(self, filepath):
        with np.load(filepath) as data:
            for i, module in enumerate(self.modules_list):
                if f'W{i}' in data:
                    module.W = data[f'W{i}']
                if hasattr(module, 'b') and f'b{i}' in data:
                    module.b = data[f'b{i}']

        return



if __name__ == '__main__':
    print('okk')
    bz = 5
    in_dim = 100
    hidden_dim = 64
    cls_dim = 3
    lr = 4e-3

    X = np.random.randn( bz, in_dim )
    Y = np.random.randint(0, cls_dim, bz)
    model = MLP_3layer_Model(in_dim,hidden_dim,cls_dim)

    log_prob , loss = model(X,labels=None)
    # print(loss)
    # print(log_prob)

    # predict
    acc = (np.argmax(log_prob,axis=-1) == Y).sum() / bz
    print('origin:')
    print(acc)
    print('---------------')
    print('start to train ...')
    # xxxx
    optim_args = {'lr':lr , 'lambda_w':1.0}
    for _ in range(20):
        model.train()
        log_prob, loss = model(X, Y)
        model.backpropagation(optim_args=optim_args)

        model.eval()
        acc = (np.argmax(log_prob, axis=-1) == Y).sum() / bz
        print(acc)


    # einsum
    def test_einsum():
        A = np.random.randn(10, 5, 6)
        B = np.random.randn(10, 6, 3)
        # 使用 einsum 执行矩阵乘法
        C = np.einsum('bij,bjk->bik', A, B)
        C == A @ B
        print(C)  # 输出将会是 A 和 B 的乘积，即 [[19, 22], [43, 50]]


        A = np.random.randn(10, 5, 6)
        B = np.random.randn(10, 6)
        # 使用 einsum 执行矩阵乘法
        C = np.einsum('bij,bj->bi', A, B)
        C[0] == A[0] @ B[0]
        C == A @ B
        print(C)


        A = np.random.randn(10,1, 5, 6)
        B = np.random.randn(10,1, 5, 6)
        # 使用 einsum 执行矩阵乘法
        C = np.einsum('bnij,bnij->bn', A, B)
        print(C[0,0])
        print( (A[0,0] * B[0,0]).sum() )
        print( C[0,0] == (A[0,0] * B[0,0]).sum()  )

        A = np.random.randn(10,7, 7, 5)
        B = np.random.randn(10,7, 5, 6)
        # 使用 einsum 执行矩阵乘法
        C = np.einsum('bnij,bnij->bn', A, B)
        print(C[0,0])
        print( (A[0,0] * B[0,0]).sum() )
        print( C[0,0] == (A[0,0] * B[0,0]).sum()  )


        A = np.random.randn(10,1, 7, 5)
        B = np.random.randn(10,1,)
        C = np.einsum('bnij,bn->bij', A,B)
        print(C[0] == B[0]*A[0,0])


        print((A[0, 0] * B[0, 0]).sum())
        print(C[0, 0] == (A[0, 0] * B[0, 0]).sum())
        return
