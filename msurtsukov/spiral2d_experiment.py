from torch import nn

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
    import pathlib

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np
    import torch
    from torch import Tensor
    from torch.autograd import Variable
    from torch.nn import functional as F

    from NeuralODE import NeuralODE
    from mysolver import simplest_euler_ode_solver as ode_solve

    parser = argparse.ArgumentParser()
    parser.add_argument('--t_max', type=float, default=6.29*5)#why?
    parser.add_argument('--n_points', type=int, default=200)
    parser.add_argument('--min_delta_time', type=float, default=1.0)
    parser.add_argument('--max_delta_time', type=float, default=5.0)
    parser.add_argument('--max_points_num', type=int, default=32)
    parser.add_argument('--plot_freq', type=int, default=10)
    parser.add_argument('--save_path', type=pathlib.Path, required=True)
    parser.add_argument('--n_steps', type=int, default=500)
    args = parser.parse_args()
    #make save_plot directory
    args.save_path.mkdir(parents=True, exist_ok=True)

    index_np = np.arange(0, args.n_points, 1, dtype=np.int)
    index_np = np.hstack([index_np[:, None]])
    times_np = np.linspace(0, args.t_max, num=args.n_points)
    times_np = np.hstack([times_np[:, None]])

#create data
    z0 = Variable(torch.Tensor([[.6, .3]]))
    #z0 = Variable(torch.randn(1, 2))
    ode_true = NeuralODE(LinearODEF(Tensor([[-0.1, -1.], [1., -0.1]])))#spiral function ODE
    ode_trained = NeuralODE(LinearODEF(torch.randn(2, 2)/2.))#Random Linear ODE

    times = torch.from_numpy(times_np[:, :, None]).to(z0)
    obs = ode_true(z0, times, return_whole_sequence=True).detach()
    obs = obs + torch.randn_like(obs) * 0.01


    def create_batch():
        t0 = np.random.uniform(0, args.t_max - args.max_delta_time)
        t1 = t0 + np.random.uniform(args.min_delta_time, args.max_delta_time)

        idx = sorted(np.random.permutation(index_np[(times_np > t0) & (times_np < t1)])[:args.max_points_num])

        obs_ = obs[idx]
        ts_ = times[idx]
        return obs_, ts_
    
    def plot_trajectories(obs=None, times=None, trajs=None, save=None, figsize=(16, 8)):
        plt.figure(figsize=figsize)
        if obs is not None:
            if times is None:
                times = [None] * len(obs)
            for o, t in zip(obs, times):
                o, t = o.detach().cpu().numpy(), t.detach().cpu().numpy()
                for b_i in range(o.shape[1]):
                    plt.scatter(o[:, b_i, 0], o[:, b_i, 1], c=t[:, b_i, 0], cmap=cm.plasma)

        if trajs is not None:
            for z in trajs:
                z = z.detach().cpu().numpy()
                plt.plot(z[:, 0, 0], z[:, 0, 1], lw=1.5)
            if save is not None:
                plt.savefig(save)
        plt.close()
        #plt.show()

#Train Neural ODE
    optimizer = torch.optim.Adam(ode_trained.parameters(), lr=0.01)
    for i in range(args.n_steps):
        obs_, ts_ = create_batch()
        z_ = ode_trained(obs_[0], ts_, return_whole_sequence=True)
        loss = F.mse_loss(z_, obs_.detach())

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        if i % args.plot_freq == 0:
            z_p = ode_trained(z0, times, return_whole_sequence=True)
            save_filename = args.save_path / 'step{:04d}.png'.format(i)
            plot_trajectories(obs=[obs], times=[times], trajs=[z_p], save=str(save_filename.resolve()))
