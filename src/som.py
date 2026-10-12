"""
Self-Organizing Map (SOM), implemented from scratch rather than via a
library (e.g. MiniSom) -- this mirrors exactly the algorithm structure
covered in the syllabus (Module 4: weight initialization, competitive
learning / best-matching-unit selection, neighbourhood function,
learning-rate and neighbourhood decay), and demonstrates the mechanics
rather than treating the SOM as a black box.

Role in this project: SOM is used here as a COMPLEMENTARY, topology-
preserving 2D view of the same customer feature space that K-Means
clustered. Where PCA gives a linear projection, SOM gives a non-linear
one that preserves neighbourhood relationships -- customers who are
behaviourally similar land on nearby map nodes even if that similarity
isn't captured by the top 2 PCA axes. We use it to visually cross-check
whether the K-Means segments occupy coherent, separated regions of the
map (a second, independent piece of evidence for the segmentation,
alongside the GMM/ARI check in clustering.py) -- NOT as the primary
segmentation method, since K-Means is the syllabus-aligned primary
technique (CO3) and SOM output (a grid position) isn't directly an
assignable "segment" the way a cluster label is.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


class SimpleSOM:
    """
    A minimal, readable Self-Organizing Map.

    Grid: grid_rows x grid_cols neurons, each holding a weight vector of
    dimension input_dim (same dimensionality as the scaled customer
    feature vectors).

    Training follows the standard SOM algorithm:
      1. For each input vector, find the Best Matching Unit (BMU) -- the
         neuron whose weight vector is closest (competitive learning).
      2. Update the BMU and its neighbours (within a shrinking radius)
         to move their weights slightly toward the input vector, with the
         update strength falling off with distance from the BMU (the
         neighbourhood function) and with training progress (learning
         rate decay).
      3. Repeat for a fixed number of iterations, shrinking both the
         neighbourhood radius and learning rate over time so the map
         coarsely organizes early and fine-tunes late.
    """

    def __init__(self, grid_rows=8, grid_cols=8, input_dim=8,
                 initial_learning_rate=0.5, initial_sigma=None,
                 random_state=42):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.input_dim = input_dim
        self.initial_learning_rate = initial_learning_rate
        # Neighbourhood radius starts covering roughly half the map
        self.initial_sigma = initial_sigma or max(grid_rows, grid_cols) / 2.0

        rng = np.random.default_rng(random_state)
        # Small random initial weights (data is standardized, so weights
        # centered near 0 with small spread are a sensible starting point)
        self.weights = rng.normal(loc=0.0, scale=0.1,
                                   size=(grid_rows, grid_cols, input_dim))

        # Precompute grid coordinates for neighbourhood-distance calculations
        rows, cols = np.meshgrid(np.arange(grid_rows), np.arange(grid_cols), indexing="ij")
        self.grid_coords = np.stack([rows, cols], axis=-1).astype(float)  # (rows, cols, 2)

    def _find_bmu(self, x):
        """Returns the (row, col) of the neuron whose weight vector is
        closest to input vector x -- this is the 'competitive learning'
        step where neurons compete and only the winner (and its
        neighbours) get updated."""
        dists = np.linalg.norm(self.weights - x, axis=-1)
        bmu_idx = np.unravel_index(np.argmin(dists), dists.shape)
        return bmu_idx

    def train(self, X, n_iterations=2000, verbose_every=500):
        n_samples = X.shape[0]
        rng = np.random.default_rng(0)

        for it in range(n_iterations):
            # Decay learning rate and neighbourhood radius over training
            progress = it / n_iterations
            lr = self.initial_learning_rate * np.exp(-progress * 3)
            sigma = self.initial_sigma * np.exp(-progress * 3)

            x = X[rng.integers(0, n_samples)]
            bmu_row, bmu_col = self._find_bmu(x)

            # Neighbourhood function: Gaussian falloff in grid distance
            # from the BMU -- neurons near the BMU move a lot toward x,
            # far neurons barely move at all.
            grid_dist_sq = np.sum((self.grid_coords - np.array([bmu_row, bmu_col]))**2, axis=-1)
            neighbourhood = np.exp(-grid_dist_sq / (2 * sigma**2 + 1e-9))

            # Weight update: move every neuron toward x, scaled by how much
            # learning rate and neighbourhood strength it currently has.
            influence = (lr * neighbourhood)[..., np.newaxis]
            self.weights += influence * (x - self.weights)

            if verbose_every and (it + 1) % verbose_every == 0:
                logger.info(f"SOM training: iteration {it+1}/{n_iterations}, "
                            f"lr={lr:.4f}, sigma={sigma:.4f}")

        logger.info("SOM training complete")
        return self

    def transform(self, X):
        """Maps each row of X to its BMU grid coordinates."""
        return np.array([self._find_bmu(x) for x in X])

    def u_matrix(self):
        """
        Computes the U-Matrix (Unified Distance Matrix): for each neuron,
        the average distance to its immediate grid neighbours. High values
        mark boundaries between distinct regions of the map -- this is the
        standard way to visualize cluster structure directly on a SOM
        grid without needing external cluster labels.
        """
        u = np.zeros((self.grid_rows, self.grid_cols))
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                neighbours = []
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.grid_rows and 0 <= nc < self.grid_cols:
                        neighbours.append(np.linalg.norm(self.weights[r, c] - self.weights[nr, nc]))
                u[r, c] = np.mean(neighbours) if neighbours else 0
        return u


if __name__ == "__main__":
    # Smoke test with random data
    logging.basicConfig(level=logging.INFO)
    X = np.random.default_rng(0).normal(size=(200, 8))
    som = SimpleSOM(grid_rows=6, grid_cols=6, input_dim=8)
    som.train(X, n_iterations=1000)
    bmus = som.transform(X[:5])
    print("Sample BMU coordinates:", bmus)
    print("U-matrix shape:", som.u_matrix().shape)
