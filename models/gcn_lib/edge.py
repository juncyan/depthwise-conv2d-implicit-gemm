import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


def pairwise_distance(x):
    """
    Compute pairwise distance of a point cloud.
    Args:
        x: tensor (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    with paddle.no_grad():
        z = paddle.transpose(x, [0,2,1])
        x_inner = -2*paddle.matmul(x, z)
        x_square = paddle.sum(paddle.multiply(x, x), axis=-1, keepdim=True)
        z = paddle.transpose(x_square, [0,2,1])
        return x_square + x_inner + z

def part_pairwise_distance(x, start_idx=0, end_idx=1):
    """
    Compute pairwise distance of a point cloud.
    Args:
        x: tensor (batch_size, num_points, num_axiss)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    with paddle.no_grad():
        x_part = x[:, start_idx:end_idx]
        x_square_part = paddle.sum(paddle.multiply(x_part, x_part), axis=-1, keepdim=True)
        z = paddle.transpose(x, [0,2,1])
        x_inner = -2*paddle.matmul(x_part, z)
        x_square = paddle.sum(paddle.multiply(x, x), axis=-1, keepdim=True)
        z = paddle.transpose(x_square, [0,2,1])
        return x_square_part + x_inner + z


def xy_pairwise_distance(x, y):
    """
    Compute pairwise distance of a point cloud.
    Args:
        x: tensor (batch_size, num_points, num_axiss)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    with paddle.no_grad():
        z = paddle.transpose(y, [0,2,1])
        xy_inner = -2*paddle.matmul(x, z)
        x_square = paddle.sum(paddle.multiply(x, x), axis=-1, keepdim=True)
        y_square = paddle.sum(paddle.multiply(y, y), axis=-1, keepdim=True)
        z = paddle.transpose(y_square, [0,2,1])
        return x_square + xy_inner + z


def dense_knn_matrix(x, k=16, relative_pos=None):
    """Get KNN based on the pairwise distance.
    Args:
        x: (batch_size, num_axiss, num_points, 1)
        k: int
    Returns:
        nearest neighbors: (batch_size, num_points, k) (batch_size, num_points, k)
    """
    with paddle.no_grad():
        # x = x.transpose(2, 1).squeeze(-1)
        x = paddle.transpose(x, [0,2,1,3]).squeeze(-1)
        batch_size, n_points, _ = x.shape
        ### memory efficient implementation ###
        n_part = 10000
        if n_points > n_part:
            nn_idx_list = []
            groups = math.ceil(n_points / n_part)
            for i in range(groups):
                start_idx = n_part * i
                end_idx = min(n_points, n_part * (i + 1))
                dist = part_pairwise_distance(x.detach(), start_idx, end_idx)
                if relative_pos is not None:
                    dist += relative_pos[:, start_idx:end_idx]
                _, nn_idx_part = paddle.topk(-dist, k=k)
                nn_idx_list += [nn_idx_part]
            nn_idx = paddle.concat(nn_idx_list, axis=1)
        else:
            dist = pairwise_distance(x.detach())
            if relative_pos is not None:
                dist += relative_pos
                
            _, nn_idx = paddle.topk(-dist, k=k) # b, n, k
        ######
        
        center_idx = paddle.arange(0, n_points)#.repeat(batch_size, k, 1)
        
        center_idx = repeat(center_idx, batch_size, k)
        
        # center_idx = paddle.transpose(center_idx, [0,2,1])
        
    return paddle.stack((nn_idx, center_idx), axis=0)

def repeat(x, batch_size, k):
    x = x.unsqueeze(axis=-1)
    x = paddle.repeat_interleave(x, k, 1)
    # print(x)
    x = x.unsqueeze(axis=0)
    x = paddle.repeat_interleave(x, batch_size, 0)
    return x

def xy_dense_knn_matrix(x, y, k=16, relative_pos=None):
    """Get KNN based on the pairwise distance.
    Args:
        x: (batch_size, num_axiss, num_points, 1)
        k: int
    Returns:
        nearest neighbors: (batch_size, num_points, k) (batch_size, num_points, k)
    """
    with paddle.no_grad():
        x = paddle.transpose(x, [0,2,1,3])
        y = paddle.transpose(y, [0,2,1,3])
        x = x.squeeze(-1)
        y = y.squeeze(-1)
        batch_size, n_points, n_axiss = x.shape
        dist = xy_pairwise_distance(x.detach(), y.detach())
        if relative_pos is not None:
            dist += relative_pos
        _, nn_idx = paddle.topk(-dist, k=k)
        center_idx = paddle.arange(0, n_points)#.repeat(batch_size, k, 1)
        center_idx = repeat(center_idx, batch_size, k)
        
        # center_idx = paddle.transpose(center_idx,[0,2,1])
        
    return paddle.stack((nn_idx, center_idx), axis=0)


class DenseDilated(nn.Layer):
    """
    Find dilated neighbor from neighbor list

    edge_index: (2, batch_size, num_points, k)
    """
    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(DenseDilated, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k

    def forward(self, edge_index):
        if self.stochastic:
            if paddle.rand(1) < self.epsilon and self.training:
                num = self.k * self.dilation
                randnum = paddle.randperm(num)[:self.k]
                edge_index = edge_index[:, :, :, randnum]
            else:
                edge_index = edge_index[:, :, :, ::self.dilation]
        else:
            edge_index = edge_index[:, :, :, ::self.dilation]
        return edge_index


class DenseDilatedKnnGraph(nn.Layer):
    """
    Find the neighbors' indices based on dilated knn
    """
    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(DenseDilatedKnnGraph, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k
        self._dilated = DenseDilated(k, dilation, stochastic, epsilon)

    def forward(self, x, y=None, relative_pos=None):
        if y is not None:
            #### normalize
            x = F.normalize(x, p=2.0, axis=1)
            y = F.normalize(y, p=2.0, axis=1)
            ####
            edge_index = xy_dense_knn_matrix(x, y, self.k * self.dilation, relative_pos)
        else:
            #### normalize
            x = F.normalize(x, p=2.0, axis=1)
            ####
            edge_index = dense_knn_matrix(x, self.k * self.dilation, relative_pos)
        return self._dilated(edge_index)