import numpy as np

from cht_tiling.utils import png2elevation, elevation2png

z = np.random.rand(256, 256) * 1000.0 - 500.0

vmin = -100.0
vmax = 100.0

options = ["float8", "float16", "float24", "float32"]

for encoder in options:
    print(f"Testing {encoder}")
    elevation2png(z, "test.png", encoder=encoder, encoder_vmin=vmin, encoder_vmax=vmax)
    z2 = png2elevation("test.png", encoder=encoder, encoder_vmin=vmin, encoder_vmax=vmax)
    print(z[0,0], z2[0,0]) 

factor=100
i = (np.random.rand(256, 256) * factor).astype(int)

options = ["uint8", "uint16", "uint24", "uint32"]

for encoder in options:
    print(f"Testing {encoder}")
    try:
        elevation2png(i, "testi.png", encoder=encoder, encoder_vmin=vmin, encoder_vmax=vmax)
        i2 = png2elevation("testi.png", encoder=encoder, encoder_vmin=vmin, encoder_vmax=vmax)
        assert np.allclose(i, i2), f"Failed {encoder}. Numbers are not the same"
        print("Success " + encoder)
    except Exception as e:
        print(f"Failed {encoder}")
        print(e)
