import numpy as np

class dataSeparate:

    def __init__(self, rate, size):
        sumr = sum(rate)
        rate = [r/sumr for r in rate]
        sizes = []
        self.indexs = []
        self.sepa_size = len(rate)
        remnant_size = size
        for i in range(self.sepa_size):
            rem_rate = remnant_size/size
            print(remnant_size, rem_rate)
            ri = min(1, rate[i]/rem_rate)
            num = int(ri*remnant_size)
            sizes.append(num)
            remnant_size -= num

        print(size, "=", sum(sizes))
        print(sizes)

        remnant_sets = set(range(size))
        for i in range(self.sepa_size):
            idxs = np.random.choice(list(remnant_sets), sizes[i], replace=False)
            self.indexs.append(idxs)
            remnant_sets -= set(idxs)
    def separate(self, data):
        ret = []
        for idx in self.indexs:
            ret.append(data[idx])

        return ret