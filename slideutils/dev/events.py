import pandas as pd
import h5py


class Events:
    def __init__(self):
        self.features = None
        self.images = None
        self.masks = None
        self.channels = None

    def read(self, inpath):
        self.features = pd.read_hdf(inpath, mode="r", key="features")
        with h5py.File(inpath, mode="r") as f:
            self.images = f["images"][:]
            self.channels = f["channels"][:]
            if "masks" in f.keys():
                self.masks = f["masks"][:]

    def write(self, outpath):
        self.features.to_hdf(outpath, mode="a", key="features")
        with h5py.File(self.path, mode="a") as f:
            f.create_dataset("images", self.images.shape, data=self.images)
            f.create_dataset(
                "channels", len(self.channels), data=self.channels
            )
            if "masks" in f.keys():
                f.create_dataset(
                    "masks",
                    self.masks.shape,
                    data=self.masks,
                    compression="gzip",
                )
