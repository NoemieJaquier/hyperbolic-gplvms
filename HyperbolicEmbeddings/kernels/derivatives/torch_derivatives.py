import torch


def diff_1st_kernel(kernel_values: torch.Tensor, X_or_Y: torch.Tensor) -> torch.Tensor:
    """
    computes the first derivative of the kernel inputs w.r.t X or Y. Derivatives w.r.t X are
    diff_1st_k[i, j] = d/dx kernel(X[i], Y[j]) for a full kernel matrix and 
    diff_1st_k[i] = d/dx kernel(X[i], Y[i]) for only the kernel diagonal

    Parameters
    -
    kernel_values: [M, N] or [M]  kernel matrix or kernel main diagonal
    X_or_Y: [M/N, D] input points for the kernel to derive for

    returns
    -
    diff_1st_kernel: [M, N, D] or [M=N, D] first derivative of the kernel input w.r.t X or Y
    """
    M, D = kernel_values.shape[0], X_or_Y.shape[1]
    ones = torch.ones(M)
    if len(kernel_values.shape) == 1:
        return torch.autograd.grad(kernel_values, X_or_Y, grad_outputs=ones, create_graph=True)[0]

    N = kernel_values.shape[1]
    diff_1st_kernel = torch.zeros(M, N, D)
    for j in range(N):
        diff_1st_kernel[:, j] = torch.autograd.grad(kernel_values[:, j], X_or_Y, grad_outputs=ones, create_graph=True)[0]
    return diff_1st_kernel


def diff_2nd_kernel(diff_1st_kernel: torch.Tensor, X_or_Y: torch.Tensor) -> torch.Tensor:
    """
    computes the second derivative of the kernel inputs w.r.t. X or Y. Assuming the first derivative is w.r.t. X and the second should be w.r.t Y then this function computes 
    diff_2nd_k[i, j] = d^2/dydx kernel(X[i], Y[j]) for a full kernel matrix and
    diff_2nd_k[i] = d^2/dydx kernel(X[i], Y[i]) for only the kernel diagonal

    Parameters
    -
    diff_1st_kernel: [M, N, D] or [M, D] kernel matrix derivative or kernel diagonal derivative w.r.t X or Y
    X_or_Y: [M/N, D] input points for the second derivative to derive for 

    returns
    -
    diff_2nd_kernel: [M, N, D, D] or [M, D, D] second derivative of the kernel input w.r.t. X or Y
    """
    M, D = diff_1st_kernel.shape[0], X_or_Y.shape[1]
    ones = torch.ones(M)
    if len(diff_1st_kernel.shape) == 2:
        diff_2nd_kernel = torch.zeros(M, D, D)
        for d in range(D):
            diff_2nd_kernel[:, d] = torch.autograd.grad(diff_1st_kernel[:, d], X_or_Y, grad_outputs=ones, create_graph=True)[0]
        return diff_2nd_kernel

    N = diff_1st_kernel.shape[1]
    diff_2nd_kernel = torch.zeros(M, N, D, D)
    for j in range(N):
        for d in range(D):
            diff_2nd_kernel[:, j, d] = torch.autograd.grad(diff_1st_kernel[:, j, d], X_or_Y,  grad_outputs=ones, create_graph=True)[0]
    return diff_2nd_kernel


def diff_3rd_kernel(diff_2nd_kernel: torch.Tensor, X_or_Y: torch.Tensor) -> torch.Tensor:
    """
    computes the third derivative of the kernel inputs w.r.t. X or Y. Assuming the first derivative is w.r.t. X, the second w.r.t Y and the third should be w.r.t Y then this function computes 
    diff_3rd_k[i, j] = d^3/dydydx kernel(X[i], Y[j]) for a full kernel matrix and
    diff_3rd_k[i] = d^3/dydydx kernel(X[i], Y[i]) for only the kernel diagonal

    Parameters
    -
    diff_2nd_kernel: [M, N, D, D] or [M, D, D] kernel matrix derivative or kernel diagonal derivative w.r.t XX, XY, or YY
    X_or_Y: [M/N, D] input points for the third derivative to derive for 

    returns
    -
    diff_2nd_kernel: [M, N, D, D, D] or [M, D, D, D] third derivative of the kernel input w.r.t. X or Y
    """
    M, D = diff_2nd_kernel.shape[0], X_or_Y.shape[1]
    ones = torch.ones(M)
    if len(diff_2nd_kernel.shape) == 3:
        diff_3rd_kernel = torch.zeros(M, D, D, D)
        for d1 in range(D):
            for d2 in range(D):
                diff_3rd_kernel[:, d1, d2] = torch.autograd.grad(diff_2nd_kernel[:, d1, d2], X_or_Y, grad_outputs=ones, create_graph=True)[0]
        return diff_3rd_kernel

    N = diff_1st_kernel.shape[1]
    diff_3rd_kernel = torch.zeros(M, N, D, D, D)
    for j in range(N):
        for d1 in range(D):
            for d2 in range(D):
                diff_3rd_kernel[:, j, d1, d2] = torch.autograd.grad(diff_2nd_kernel[:, j, d1, d2], X_or_Y,  grad_outputs=ones, create_graph=True)[0]
    return diff_3rd_kernel
