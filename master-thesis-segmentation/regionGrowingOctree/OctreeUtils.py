def get_morton_index(x, y, z, depth: int):
    # Shift and mask bits for each dimension
    x = (x >> depth) & 1
    y = (y >> depth) & 1
    z = (z >> depth) & 1
    # Interleave the bits
    return interleave_bits(x, y, z)


def interleave_bits(x, y, z):
    result = 0
    for i in range(10):  # Assuming 10 bits are used for each dimension
        result |= (x & 1) << (3 * i)
        result |= (y & 1) << (3 * i + 1)
        result |= (z & 1) << (3 * i + 2)
        x >>= 1
        y >>= 1
        z >>= 1
    return result
