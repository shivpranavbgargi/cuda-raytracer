import numpy as np

def load_ppm(path):
    with open(path, "rb") as f:
        assert f.readline().strip() == b"P6"
        line = f.readline()
        while line.startswith(b"#"):      # skip comments
            line = f.readline()
        w, h = map(int, line.split())
        maxval = int(f.readline())
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(h, w, 3).astype(int)

a = load_ppm("gpu_image.ppm.v1")
b = load_ppm("gpu_image.ppm")
d = np.abs(a - b)
print("max diff:", d.max())
print("differing pixels:", d.any(axis=-1).mean())
