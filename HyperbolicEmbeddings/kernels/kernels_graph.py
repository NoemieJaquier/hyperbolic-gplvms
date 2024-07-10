import torch
import gpytorch
import networkx as nx


# if torch.cuda.is_available():
#     device = torch.cuda.current_device()
# else:
#     device = 'cpu'
device = 'cpu'


class GraphGaussianKernel(gpytorch.kernels.Kernel):
    """
    Instances of this class represent a Gaussian (RBF) covariance matrix between input points on a graph.

    Attributes
    ----------
    self.adjacency_matrix: graph adjacency matrix
    self.nu: smoothness parameter
    self.eigenvalues: eigenvalues of the graph Laplacian
    self.eigenvectors: eigenvectors of the graph Laplacians
    self.num_verticies: number of vertices of the graph

    Methods
    -------
    eigenvalues_function()
    forward(x1_id, x2_id, diagonal_matrix_flag=False, **params)

    References
    ----------
    [1] Borovitskiy, V. et al. Matern Gaussian Processes on Graphs. In AISTATS, 2021.

    """
    def __init__(self, adjacency_matrix, eigenvalues=None, eigenvectors=None, num_eigenpairs=None, **kwargs):
        """
        Initialisation.

        Parameters
        ----------
        :param adjacency_matrix: graph adjacency matrix

        Optional parameters
        -------------------
        :param nu: smoothness parameter
        :param eigenvalues: eigenvalues of the graph Laplacian
        :param eigenvectors: eigenvectors of the graph Laplacian
        :param num_eigenpairs: number of eigenpairs to consider for the kernel computation
        """
        self.has_lengthscale = True
        super(GraphGaussianKernel, self).__init__(has_lengthscale=True, ard_num_dims=None, **kwargs)

        self.adjacency_matrix = adjacency_matrix

        if eigenvalues and eigenvectors:
            # Use given eigenvalues and eigenvectors
            self.eigenvectors = eigenvectors
            self.eigenvalues = eigenvalues
        else:
            # Compute eigenvalues and eigenvectors of the graph Laplacian
            graph = nx.from_numpy_array(self.adjacency_matrix)
            laplacian = torch.from_numpy(nx.laplacian_matrix(graph).toarray())
            self.eigenvalues, self.eigenvectors = torch.linalg.eigh(laplacian)

        # Reduce number of eigenpairs used to compute the kernel
        if num_eigenpairs:
            if num_eigenpairs > self.eigenvectors.shape[0]:
                num_eigenpairs = self.eigenvectors.shape[0]
            self.eigenvectors = self.eigenvectors[:, :num_eigenpairs]
            self.eigenvalues = self.eigenvalues[:num_eigenpairs]

        self.num_verticies = self.eigenvectors.shape[0]

    def eigenvalues_function(self):
        """
        Apply the function leading to the Gaussian kernel on the eigenvalues of the adjacency matrix of the graph.

        Return
        ------
        :return f(eigenvalue) = exp (lengthscale^2 / 4 * eigenvalues)
        """
        S = torch.exp(self.lengthscale**2 / 4. * self.eigenvalues)  # Gaussian
        S = torch.multiply(S, self.num_verticies / torch.sum(S))
        return S

    def forward(self, x1_id, x2_id, diag=False, **params):
        """
        Computes the Gaussian kernel matrix given the provided matrix of graph distances

        Parameters
        ----------
        :param x1_id:
        :param x2_id:

        Optional parameters
        -------------------
        :param diag: Should we return the whole distance matrix, or just the diagonal?
        :param params: additional parameters

        Returns
        -------
        :return: kernel matrix corresponding to the graph distance
        """
        if x1_id.ndim == 2:
            x1_id = x1_id[:, 0]

        if x2_id.ndim == 2:
            x2_id = x2_id[:, 0]

        # Compute function of eigenvalues
        f_eigs = self.eigenvalues_function()[0]

        # Kernel = eigenvector * f(eigenvalues) * eigenvector.T
        eigvecs1 = self.eigenvectors[x1_id, :]
        eigvecs2 = self.eigenvectors[x2_id, :]
        kernel = torch.matmul(torch.matmul(eigvecs1, torch.diag(f_eigs)), eigvecs2.T)

        return kernel


class GraphMaternKernel(gpytorch.kernels.Kernel):
    """
    Instances of this class represent a Matérn covariance matrix between input points on a graph.

    Attributes
    ----------
    self.adjacency_matrix: graph adjacency matrix
    self.nu: smoothness parameter
    self.eigenvalues: eigenvalues of the graph Laplacian
    self.eigenvectors: eigenvectors of the graph Laplacians
    self.num_verticies: number of vertices of the graph

    Methods
    -------
    eigenvalues_function()
    forward(x1_id, x2_id, diagonal_matrix_flag=False, **params)

    References
    ----------
    [1] Borovitskiy, V. et al. Matern Gaussian Processes on Graphs. In AISTATS, 2021.

    """
    def __init__(self, adjacency_matrix, nu=2.5, eigenvalues=None, eigenvectors=None, num_eigenpairs=None, **kwargs):
        """
        Initialisation.

        Parameters
        ----------
        :param adjacency_matrix: graph adjacency matrix

        Optional parameters
        -------------------
        :param nu: smoothness parameter
        :param eigenvalues: eigenvalues of the graph Laplacian
        :param eigenvectors: eigenvectors of the graph Laplacian
        :param num_eigenpairs: number of eigenpairs to consider for the kernel computation
        :param kwargs: additional arguments
        """
        self.has_lengthscale = True
        super(GraphMaternKernel, self).__init__(has_lengthscale=True, ard_num_dims=None, **kwargs)

        self.nu = nu

        self.adjacency_matrix = adjacency_matrix

        if eigenvalues and eigenvectors:
            # Use given eigenvalues and eigenvectors
            self.eigenvectors = eigenvectors
            self.eigenvalues = eigenvalues
        else:
            # Compute eigenvalues and eigenvectors of the graph Laplacian
            graph = nx.from_numpy_array(self.adjacency_matrix)
            laplacian = torch.from_numpy(nx.laplacian_matrix(graph).toarray())
            self.eigenvalues, self.eigenvectors = torch.linalg.eigh(laplacian)

        # Reduce number of eigenpairs used to compute the kernel
        if num_eigenpairs:
            if num_eigenpairs > self.eigenvectors.shape[0]:
                num_eigenpairs = self.eigenvectors.shape[0]
            self.eigenvectors = self.eigenvectors[:, :num_eigenpairs]
            self.eigenvalues = self.eigenvalues[:num_eigenpairs]

        self.num_verticies = self.eigenvectors.shape[0]

    def eigenvalues_function(self):
        """
        Apply the function leading to the Gaussian kernel on the eigenvalues of the adjacency matrix of the graph.

        Return
        ------
        :return f(eigenvalue) = (2 nu / lengthscale^2 + eigenvalues)^(nu/2)
        """
        S = torch.pow(self.eigenvalues + 2 * self.nu / self.lengthscale**2, -self.nu)  # Matern
        S = torch.multiply(S, self.num_verticies / torch.sum(S))
        return S

    def forward(self, x1_id, x2_id, diag=False, **params):
        """
        Computes the graph Matern kernel matrix.

        Parameters
        ----------
        :param x1_id:
        :param x2_id:

        Optional parameters
        -------------------
        :param diag: Should we return the whole distance matrix, or just the diagonal?
        :param params: additional parameters

        Returns
        -------
        :return: kernel matrix corresponding to the graph distance
        """
        if x1_id.ndim == 2:
            x1_id = x1_id[:, 0]

        if x2_id.ndim == 2:
            x2_id = x2_id[:, 0]

        # Compute function of eigenvalues
        f_eigs = self.eigenvalues_function()[0]

        # Kernel = eigenvector * f(eigenvalues) * eigenvector.T
        eigvecs1 = self.eigenvectors[x1_id, :]
        eigvecs2 = self.eigenvectors[x2_id, :]
        kernel = torch.matmul(torch.matmul(eigvecs1, torch.diag(f_eigs)), eigvecs2.T)

        return kernel

