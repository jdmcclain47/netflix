"""
serialize the data into binary format for fast processing

example:

    In [2]: %time np.fromfile("all.dta.bin", dtype = np.int32).reshape(-1, 4)
    CPU times: user 288 µs, sys: 1.9 s, total: 1.91 s
    Wall time: 2.31 s
    Out[2]:
    array([[   400,      1,   1828,      2],
           [  1110,      1,   1737,      3],
           [  1370,      1,   2056,      3],
           ...,
           [457557,  17770,   2180,      4],
           [457874,  17770,   1617,      3],
           [458137,  17770,   2107,      5]], dtype=int32)

    In [3]: %time np.fromfile("all.idx.bin", dtype = np.int32)
    CPU times: user 348 µs, sys: 539 ms, total: 539 ms
    Wall time: 664 ms
    Out[3]: array([1, 1, 1, ..., 1, 1, 1], dtype=int32)

    In [4]: %time np.fromfile("qual.dta.bin", dtype = np.int32)
    CPU times: user 322 µs, sys: 39.4 ms, total: 39.8 ms
    Wall time: 47 ms
    Out[4]: array([ 29674,      1,   1986, ..., 452990,  17770,   2172], dtype=int32)

"""


import struct


tobin = [("data/mu/all.dta", 4 * "i"), ("data/mu/all.idx", "i"), ("data/mu/qual.dta", 3 * "i")]

size = 2048

for fname, dtype in tobin:
    with open(fname + "_np", "wb") as fbin:
        pass

    i = 0
    with open(fname, "r") as f:
        buf = []
        line = f.readline()
        count = 0
        while line:
            buf += map(int, line.split())
            count += 1
            if count == size:
                i += 1
                with open(fname + ".bin", "ab") as fbin:
                    fbin.write(struct.pack(size * dtype, *buf))
                buf = []
                count = 0
                if i % 100 == 0:
                    print i * size
            line = f.readline()

        with open(fname + "_np", "ab") as fbin:
            fbin.write(struct.pack(count * dtype, *buf))
