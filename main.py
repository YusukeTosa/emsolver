import os
import numpy as np
from PIL import Image


class EMSolver():
    def __init__(self, M, sigma):
        self.M = M
        self.sigma = sigma

    def calc_posterior(self, mu, i):
        def gaussian(x, mu=mu, i=i):
            return np.exp(-1/(2 * self.sigma**2)*np.linalg.norm(x - mu[i])**2)

        def wrapper(x, mu=mu, i=i):
            return gaussian(x, mu, i) / np.sum([gaussian(x, mu, j) for j in range(self.M)])
        return wrapper

    def calc_expectation(self, mu, i):
        def gaussian(x, mu=mu, i=i):
            return np.exp(-1/(2 * self.sigma**2)*np.linalg.norm(x - mu[i])**2)

        def wrapper(x, mu=mu, i=i):
            return x * (gaussian(x, mu, i) / np.sum([gaussian(x, mu, j) for j in range(self.M)]))
        return wrapper

    def load(self, csv_data):
        data = np.loadtxt(csv_data, delimiter=",", skiprows=1)/256
        init_mu = data[np.random.choice(data.shape[0], self.M, replace=False), :]
        return data, init_mu

    def e_step(self, data, mu):
        stats = []
        for i in range(self.M):
            partition = np.mean(np.apply_along_axis(self.calc_posterior(mu=mu, i=i), 1, data))
            expectation = np.mean(np.apply_along_axis(self.calc_expectation(mu=mu, i=i), 1, data), axis=0)
            stats.append((expectation, partition))
        return stats

    def m_step(self, stats):
        mu = np.array([e/p for e, p in stats])
        return mu


def main():
    M = 3
    sigma = 1.0
    solver = EMSolver(M, sigma)
    print('loading csv')
    data, mu = solver.load("mnist_em.csv")
    delta_mu = np.full(M, np.inf)
    while np.any(delta_mu > 0.01):
        mu_old = mu
        stats = solver.e_step(data, mu)
        mu = solver.m_step(stats)
        delta_mu = np.sum(np.square(mu - mu_old), axis=1)
        print(delta_mu)
    print("finish")

    label = "abcdefghi"
    if M > len(label):
        label += ''.join([str(i) for i in range(M - len(label))])

    output_dir = "output_fig"

    files = os.listdir(output_dir)
    for f in files:
        os.remove(os.path.join(output_dir, f))

    for i, item in enumerate(mu):
        item = np.round(item*256).astype(np.uint8)
        item = item.reshape((28, 28))
        img = Image.fromarray(item, 'L')
        img.save(output_dir + "/mu_" + label[i] + ".png")


if __name__ == "__main__":
    main()
