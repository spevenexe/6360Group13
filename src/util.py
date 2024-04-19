import numpy as np
class Util:
    @staticmethod
    def gen_random(rng, size, N):
        arr = np.sort(rng.choice(np.arange(N), size, replace=False))
        return (arr + rng.randint(N)) % N