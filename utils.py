import numpy as np
from tqdm.auto import trange

class kMeans():
    def __init__(self, num_cluster=5, data=None):
        self.codebooks = 3*np.random.rand(num_cluster, 2)-1.5
        if data is not None:
            self.init_codes(data)

    def init_codes(self, data):
        self.codebooks = data[np.random.permutation(len(data))][:len(self.codebooks)]

    def update(self, x):
        out = self.pred(x)
        for c in range(len(self.codebooks)):
            if len(x[out==c])==0:
                continue
            self.codebooks[c] = np.mean(x[out==c], axis=0)

    def pred(self, x):
        dists = np.sqrt(np.sum((x[:,None]-self.codebooks[None])**2, axis=-1)).T
        return np.argmin(dists, axis=0)

    def fit(self, x, verbose=False, max_it=100):
        self.init_codes(x)
        alt = 0
        for i in trange(max_it):
            alt = self.codebooks.copy()
            self.update(x)
            dists = np.sum((alt-self.codebooks)**2, axis=-1)
            if np.max(dists)==0:
                break
        if verbose:
            print(f"{len(self.codebooks)} Zentren wurden in {i+1} Schritten gelernt. Fehler: {self.fehler(x)}")

    def fehler(self, x):
        dists = np.sqrt(np.sum((x[:,None]-self.codebooks[None])**2, axis=-1)).T
        d = np.argmin(dists, axis=0)
        out = 0
        for j,i in enumerate(d):
            out += dists[i,j]
        return out/len(x)