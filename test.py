import os
import glob

bdir = r"output/"
sdirs = os.listdir(bdir)
for sdir in sdirs:
    bspath = os.path.join(bdir, sdir)
    ssdirs = os.listdir(bspath)
    for ssd in ssdirs:
        pth = os.path.join(bspath, ssd)
        file_names = glob.glob(os.path.join(pth, '*.pdparams'))
        for fn in file_names:
            if not "best" in fn:
                os.remove(fn)