from ODEF import ODEF

class LinearODEF(ODEF):
    def __init__(self, W):
        super(LinearODEF, self).__init__()
        self.lin = nn.Linear(2, 2, bias=False)
        self.lin.weight = nn.Parameter(W)

    def forward(self, x, t):
        return self.lin(x)


if __name__ == "__main__":
    import argparse

    import numpy
    import torch
    from torch import Tensor
    from torch import Variable

    from NeuralODE import NeuralODE

    parser = argparse.ArgumentParser()
    parser.add_argument('--t_max', type=float, default=6.29*5)#why?
    parser.add_argument('--n_points', type=int, default=200)
    parser.add_argument('--min_delta_tyme', type=float, default=1.0)
    parser.add_argument('--max_delta_tyme', type=float, default=5.0)
    parser.add_argument('--max_points_num', type=int, default=32)
    def create_batch():
        t0 = numpy.random.uniform
    


    ode_true = NeuralODE(LinearODEF(Tensor([[-0.1, -1.], [1., -0.1]])))#spiral function ODE
    ode_trained = NeuralODE(LinearODEF(torch.randn(2, 2)/2.))#Random Linear ODE

    z0 = Variable(torch.Tensor([[.6, .3]]))
    #z0 = Variable(torch.randn(1, 2))