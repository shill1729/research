import numpy as np

from ae.symbolic.diffgeo import RiemannianManifold
from ae.toydata.pointclouds import PointCloud
from ae.toydata.surfaces import SurfaceBase
from ae.toydata.local_dynamics import DynamicsBase
from ae.utils import process_data, embed_data, random_rotation_matrix


class ToyData:
    """

    """
    def __init__(self, surface: SurfaceBase, dynamics: DynamicsBase, large_dim=None, embedding_seed=None):
        """

        :param surface:
        :param dynamics:
        """
        self.embedding_seed = embedding_seed
        self.random_embedding_matrix = None
        self.large_dim = large_dim
        self.surface = surface
        self.dynamics = dynamics
        self.manifold = self.__initialize_manifold()
        self.local_drift = self.__initialize_local_drift()
        self.local_diffusion = self.__initialize_local_diffusion()
        self.point_cloud = self.__get_point_cloud()

    def __initialize_manifold(self):
        """

        :return:
        """
        manifold = RiemannianManifold(self.surface.local_coords(), self.surface.equation())
        return manifold

    def __initialize_local_drift(self):
        """

        :return:
        """
        return self.dynamics.drift(self.manifold)

    def __initialize_local_diffusion(self):
        """

        :return:
        """
        return self.dynamics.diffusion(self.manifold)

    def __get_point_cloud(self, epsilon=0., compute_proj=True):
        """

        :param epsilon: defaults to zero for training region
        :param compute_proj: boolean for returning the orthogonal projection or not
        :return:
        """
        max_bounds = [(b[0] - epsilon, b[1] + epsilon) for b in self.surface.bounds()]
        point_cloud = PointCloud(manifold=self.manifold,
                                 bounds=max_bounds,
                                 local_drift=self.local_drift,
                                 local_diffusion=self.local_diffusion,
                                 compute_orthogonal_proj=compute_proj)
        return point_cloud

    def set_point_cloud(self, epsilon=0., compute_proj=True):
        """

        :param epsilon:
        :param compute_proj:
        :return:
        """
        self.point_cloud = self.__get_point_cloud(epsilon, compute_proj)

    def generate_data(self, n, d, seed=None, device="cpu", embed=False):
        """
        Generate and preprocess data. Make sure to use '.set_point_cloud' if you want to change the boundary,
        i.e. for testing beyond the training region.

        :param n:
        :param d:
        :param seed:
        :param device:
        :param embed:
        :return:
        """
        x, _, mu, cov, local_x = self.point_cloud.generate(n, seed=seed)
        if embed:
            print(self.embedding_seed)
            R = random_rotation_matrix(D=self.large_dim, seed=self.embedding_seed)
            self.random_embedding_matrix = R

            x, mu, cov = embed_data(x, mu, cov, R)
        x, mu, cov, p, _, orthonormal_frame = process_data(x, mu, cov, d, True, device)
        print(x.size())
        print(mu.size())
        print(cov.size())
        print(p.size())
        print(orthonormal_frame.size())
        return {
            "x": x, "mu": mu, "cov": cov, "p": p,
            "orthonormal_frame": orthonormal_frame, "local_x": local_x
        }

    def ground_truth_ensemble(self, x0, tn, ntime, npaths):
        """

        :param x0:
        :param tn:
        :param ntime:
        :param npaths:
        :return:

        Tuple of (ambient_paths, latent_paths), each of shape (npaths, ntime+1, x0.shape)
        """
        # Assuming toy data of the form (x, y, f(x,y)):
        # TODO consider implementing phi^{-1} in diff_geo.py to avoid this limitation
        d = self.manifold.local_coordinates.shape[0]
        z0_true = x0[:d]
        # TODO: currently only hypersurfaces are implemented
        ambient_paths = np.zeros((npaths, ntime + 1, d+1))
        latent_paths = self.point_cloud.latent_sde.sample_ensemble(z0_true, tn, ntime, npaths)
        for j in range(npaths):
            for i in range(ntime + 1):
                ambient_paths[j, i, :] = np.squeeze(self.point_cloud.np_phi(*latent_paths[j, i, :]))

        return ambient_paths, latent_paths
