import numpy as np, pathlib

IN  = pathlib.Path("data.ply")
OUT = pathlib.Path("data_fixed.ply")

# read header lines
with IN.open() as f:
    header = []
    while True:
        line = f.readline()
        header.append(line)
        if line.strip() == "end_header":
            break
    body = f.read()

# load whole vertex table as floats/ints
verts = np.genfromtxt(IN, skip_header=len(header), dtype=float)

# clamp last 3 columns to 0-255 and cast to uint8
verts[:, 3:6] = np.clip(verts[:, 3:6], 0, 255)
verts[:, 3:6] = verts[:, 3:6].astype(np.uint8)

# write back
with OUT.open("w") as f:
    f.writelines(header)
    np.savetxt(f, verts, fmt="%f %f %f %d %d %d")
print("Wrote", OUT)
