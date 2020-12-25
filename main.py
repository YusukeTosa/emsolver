import argparse
import logging
import os
import numpy as np
from PIL import Image

formatter = '%(levelname)s : %(asctime)s : %(message)s'
logging.basicConfig(level=logging.DEBUG, format=formatter)


class EMSolver():
    def __init__(self, M, sigma):
        self.M = M
        self.sigma = sigma

    def _calc_posterior(self, i):
        def quadra(norm):
            return -norm/(2 * self.sigma**2 + 1e-10)

        def wrapper(x, i=i):
            norms = [np.linalg.norm(x - self.mu[j])**2 for j in range(self.M)]
            norm = norms[i]
            min_norm = np.min(norms)
            return np.exp(quadra(norm) - quadra(min_norm) - np.log(np.sum(np.exp(quadra(norms - min_norm)))))
        return wrapper

    def _calc_expectation(self, i):
        def quadra(norm):
            return -norm/(2 * self.sigma**2 + 1e-10)

        def wrapper(x, i=i):
            norms = [np.linalg.norm(x - self.mu[j])**2 for j in range(self.M)]
            norm = norms[i]
            min_norm = np.min(norms)
            return x * (np.exp(quadra(norm) - quadra(min_norm) - np.log(np.sum(np.exp(quadra(norms - min_norm))))))
        return wrapper

    def load(self, csv_data):
        data = np.loadtxt(csv_data, delimiter=",", skiprows=1)/256
        self.mu = data[np.random.choice(data.shape[0], self.M, replace=False), :]
        return data

    def e_step(self, data):
        stats = []
        for i in range(self.M):
            partition = np.mean(np.apply_along_axis(self._calc_posterior(i=i), 1, data))
            expectation = np.mean(np.apply_along_axis(self._calc_expectation(i=i), 1, data), axis=0)
            stats.append((expectation, partition))
        return stats

    def m_step(self, stats):
        self.mu = np.array([e/p for e, p in stats])
        return self.mu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-M", type=int, default=3, help="Number of categories")
    parser.add_argument("-S", "--sigma", type=float, default=1.0, help="Variance of normal distribution")
    args = parser.parse_args()

    solver = EMSolver(args.M, args.sigma)
    logging.info('loading csv')
    data = solver.load("datasets/mnist_em.csv")
    delta_mu = np.full(solver.M, np.inf)
    counter = 0
    while np.any(delta_mu > 0.01):
        mu_old = solver.mu
        stats = solver.e_step(data)
        mu_new = solver.m_step(stats)
        delta_mu = np.sum(np.square(mu_new - mu_old), axis=1)
        logging.debug(f'delta_mu epoch{counter} {delta_mu}')
        counter += 1

    label = "abcdefghi"
    if solver.M > len(label):
        label += ''.join([str(i) for i in range(solver.M - len(label))])

    output_dir = "output_fig"

    files = os.listdir(output_dir)
    for f in files:
        os.remove(os.path.join(output_dir, f))

    for i, item in enumerate(solver.mu):
        item = np.round(item*256).astype(np.uint8)
        item = item.reshape((28, 28))
        img = Image.fromarray(item, 'L')
        img.save(output_dir + "/mu_" + label[i] + ".png")

    logging.info('finish')


if __name__ == "__main__":
    main()
