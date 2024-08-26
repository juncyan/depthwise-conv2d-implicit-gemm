import paddle
from paddle import nn
import paddle.nn.functional as F
import math
from typing import List, Union


class KANLinear(nn.Layer):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.05,
        scale_base=0.8,
        scale_spline=0.8,
        enable_standalone_scale_spline=True,
        base_activation=nn.Silu,
        grid_eps=0.008,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                paddle.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand([in_features, -1])
        )
        self.register_buffer("grid", grid)

        # base_weight = paddle.create_parameter(shape=[out_features, in_features],dtype=paddle.get_default_dtype())
        self.base_weight = paddle.create_parameter(shape=[in_features, out_features],dtype=paddle.get_default_dtype())
        self.spline_weight = paddle.create_parameter(shape=[in_features, out_features, grid_size + spline_order],dtype=paddle.get_default_dtype())
        # paddle.nn.Parameter(
        #     paddle.Tensor(out_features, in_features, grid_size + spline_order)
        # )
        if enable_standalone_scale_spline:
            self.spline_scaler = paddle.create_parameter(shape=[in_features, out_features],dtype=paddle.get_default_dtype())
            # paddle.nn.Parameter(
            #     paddle.Tensor(out_features, in_features)
            # )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        KMU = nn.initializer.KaimingUniform(negative_slope=math.sqrt(5) * self.scale_base,nonlinearity='leaky_relu')
        KMU(self.base_weight)
        # print(self.base_weight)
        # paddle.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with paddle.no_grad():
            noise = (
                (
                    paddle.rand([self.grid_size + 1, self.in_features, self.out_features])
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            # print("noise", noise.shape)
            t = self.curve2coeff(self.grid.T[self.spline_order : -self.spline_order],noise)
            # print("in_features", self.in_features, "out_features", self.out_features)
            # print(t.shape)
            spw_value = (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)* t
            # print(spw_value.shape, self.spline_weight.shape)
            self.spline_weight.set_value(spw_value)
            
            if self.enable_standalone_scale_spline:
                # paddle.nn.init.constant_(self.spline_scaler, self.scale_spline)
                # paddle.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)
                KMU_ = nn.initializer.KaimingUniform(negative_slope=math.sqrt(5) * self.scale_spline,nonlinearity='leaky_relu')
                KMU_(self.spline_scaler)

    def b_splines(self, x: paddle.tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (paddle.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            paddle.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert len(x.shape) == 2 and x.shape[1] == self.in_features
        grid: paddle.tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:]))
        bases = paddle.to_tensor(bases, dtype=x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.shape == [
            x.shape[0],
            self.in_features,
            self.grid_size + self.spline_order,
        ]
        return bases

    def curve2coeff(self, x: paddle.tensor, y: paddle.tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (paddle.Tensor): Input tensor of shape (batch_size, in_features).
            y (paddle.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            paddle.Tensor: Coefficients tensor of shape (in_features, out_features,  grid_size + spline_order).
        """
        # print(x.shape, self.in_features, self.out_features)
        assert len(x.shape) == 2 and x.shape[1] == self.in_features
        # print(y.shape)
        # print(x.shape[0], self.in_features, self.out_features)
        assert y.shape == [x.shape[0], self.in_features, self.out_features]
        
        A = self.b_splines(x).transpose([1, 0, 2])  
        # (in_features, batch_size, grid_size + spline_order)

        B = y.transpose([1, 0, 2])  
        # (in_features, batch_size, out_features)
        solution = paddle.linalg.lstsq(A, B)[0] 
         # (in_features, grid_size + spline_order, out_features)

        result = solution.transpose((0, 2, 1))  
        # (out_features, in_features, grid_size + spline_order)
        # print(result.shape, self.out_features, self.in_features, self.grid_size + self.spline_order)
        assert result.shape == [
            self.in_features,
            self.out_features,
            self.grid_size + self.spline_order]
        return result

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: paddle.tensor):
        assert x.shape[-1] == self.in_features
        original_shape = x.shape
        x = x.reshape([-1, self.in_features])
        # print(self.base_activation(x).shape, self.base_weight.shape)
        # self.base_weight = self.base_weight.transpose([1,0])
        # print(z)
        base_output = F.linear(self.base_activation(x), self.base_weight)
        # t1 = self.b_splines(x).reshape([x.shape[0], -1])
        # t2 = self.scaled_spline_weight.reshape([self.out_features, -1])
        # print("t1", t1.shape, "t2", t2.shape)
        spline_output = F.linear(
            self.b_splines(x).reshape([x.shape[0], -1]),
            self.scaled_spline_weight.reshape([-1,self.out_features]),
        )
        output = base_output + spline_output
        
        output = output.reshape([*original_shape[:-1], self.out_features])
        return output

    @paddle.no_grad()
    def update_grid(self, x: paddle.tensor, margin=0.01):
        print(x.shape, self.in_features)
        assert len(x.shape) == 2 and x.shape[2] == self.in_features
        batch = x.shape[0]

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.transpose([1, 0, 2])  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.transpose([1, 2, 0])  # (in, coeff, out)
        unreduced_spline_output = paddle.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.transpose([1, 0, 2])  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = paddle.sort(x, axis=0)[0]
        grid_adaptive = x_sorted[
            paddle.linspace(
                0, batch - 1, self.grid_size + 1, dtype=paddle.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            paddle.arange(
                self.grid_size + 1, dtype=paddle.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = paddle.concatenate(
            [
                grid[:1]
                - uniform_step
                * paddle.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * paddle.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            axis=0,
        )

        self.grid.set_value(grid.T)
        self.spline_weight.set_value(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -paddle.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class KAN(nn.Layer):
    def __init__(
        self,
        layers_hidden:List[Union[int, float]]=[16, 32, 64],
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=nn.Silu,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = nn.LayerList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: paddle.tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )