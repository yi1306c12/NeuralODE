import math

def simplest_euler_ode_solver(z0, t0, t1, f):
    """
    Simplest Euler ODE initial value solver
    """
    h_max = 0.05
    n_steps = math.ceil((abs(t1 - t0)/h_max).max().item())

    h = (t1 - t0)/n_steps
    t = t0
    z = z0

    for i_step in range(n_steps):
        z = z + h * f(z, t)
        t = t + h
    return z

if __name__ == "__main__":
    import math
    import torch
    z0 = torch.tensor([0., 1., 2.])
    t = torch.tensor([0., 1.])
    f = lambda x, t : x
    answer = math.e * z0
    print(answer)
    print(simplest_euler_ode_solver(z0, *t, f))