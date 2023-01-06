"""Contains all helper functions to create and load point cloud data."""

import os
import random
import numpy as np
import laspy
import pandas as pd
import torch
from typing import Dict, List

TREE_SPECIES = ['european_larch', 
                'sassafras', 
                'ca_black_oak', 
                'lombardy_poplar', 
                'eastern_cottonwood', 
                'weeping_willow', 
                'fanpalm', 
                'quaking_aspen', 
                'black_tupelo', 
                'tamarack']

def load_las_file_to_numpy(file: str) -> np.ndarray:
    return laspy.read(file).xyz.astype("float64")


class PointCloudDataSet():
    """Map-style dataset for lidar data. Contains all files in datapath ending with .las and their labels."""

    def __init__(self, datapath: str, train=True, transform=None) -> None:
        """Initialize self.all_files and self.all_labels

        Args:
            datapath (str): Folder which contains all the .las files and a .csv with the labels.
            train (bool, optional): Whether the train folder should be taken. Defaults to True.
            transform (_type_, optional): Any torch transformation(s) applied when getting the data. Defaults to None.
        """
        self.transform = transform
        self.all_files = []
        self.all_labels = []
        for root, dirs, files in os.walk(datapath):
            if train:
                dirs[:] = [d for d in dirs if "train" in d]
            else:
                dirs[:] = [d for d in dirs if "test" in d]
            
            for f in files:
                f_full_path = os.path.join(root, f)
                if f.endswith("las"):
                    self.all_files.append(f_full_path)

                if f.endswith(".csv"):
                    csv = pd.read_csv(f_full_path)
                    self.all_labels = csv

    def __len__(self) -> int:
        return len(self.all_files)

    def __getitem__(self, idx: int) -> Dict:
        try:
            points = load_las_file_to_numpy(self.all_files[idx])
            if self.transform:
                points = self.transform(points)
            file_id = int(self.all_files[idx].split('/')[-1].split('_')[-1].split('.')[0])
            label = self.all_labels.iloc[file_id, 1:].species
        except:
            print(
                f"Problem with reading file number {idx} at {self.all_files[idx]}")
            print(f"The shape of the point cloud: {points.shape}")
            raise
        return {"points": points, "label": label}


class SamplePoints():
    """Sample a subset of all points with specified methods."""

    def __init__(self, sampling_to_size: int, sample_method="random") -> None:
        """Init the sampler.

        Args:
            sampling_to_size (int): The number of points contained in the returned sample.
            sample_method (str, optional): How the sampling should be performed. One of ["random", "farthest_points"]. Defaults to "random".
        """
        assert isinstance(sampling_to_size, int)
        self.possible_sampling_strategies = ["random", "farthest_points"]
        assert sample_method in self.possible_sampling_strategies, f"Strategy {sample_method} is not in sample methods {self.possible_sampling_strategies}!"

        self.sampling_to_size = sampling_to_size
        self.sample_method = sample_method

    def _sample_randomly(self, points: torch.Tensor) -> torch.Tensor:
        """Return random sample."""
        num_points = points.shape[0]
        random_idx = random.sample(range(0, num_points), self.sampling_to_size)
        return points[random_idx, :]

    def _farthest_points(self, points: torch.Tensor) -> torch.Tensor:
        """Return farthest points sampled. Always includes and start at point at index 0. 
        The method is copied from https://minibatchai.com/sampling/2021/08/07/FPS.html.
        Note that this just serves an illustration; to speed things up, use the cuda implementation and store the sampled results."""

        # Represent the points by their indices in points
        points_left = np.arange(len(points))  # [P]

        # Initialise an array for the sampled indices
        sample_inds = np.zeros(self.sampling_to_size, dtype='int')  # [S]

        # Initialise distances to inf
        dists = np.ones_like(points_left) * float('inf')  # [P]

        # Select a point from points by its index, save it
        selected = 0
        sample_inds[0] = points_left[selected]

        # Delete selected
        points_left = np.delete(points_left, selected)  # [P - 1]

        # Iteratively select points for a maximum of self.sampling_to_size
        for i in range(1, self.sampling_to_size):
            # Find the distance to the last added point in selected
            # and all the others
            last_added = sample_inds[i-1]

            dist_to_last_added_point = (
                (points[last_added] - points[points_left])**2).sum(-1)  # [P - i]

            # If closer, updated distances
            dists[points_left] = np.minimum(dist_to_last_added_point,
                                            dists[points_left])  # [P - i]

            # We want to pick the one that has the largest nearest neighbour
            # distance to the sampled points
            selected = np.argmax(dists[points_left])
            sample_inds[i] = points_left[selected]

            # Update points_left
            points_left = np.delete(points_left, selected)

        return points[sample_inds]

    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        """Return sampled points."""
        if self.sample_method == "random":
            points = self._sample_randomly(points)
        elif self.sample_method == "farthest_points":
            points = self._sample_randomly(points)
        else:
            raise ValueError(
                f"The sampling method at initialization has to be one of {self.possible_sampling_strategies}")
        return points


def classes_to_tensor(class_names: List[str]) -> torch.Tensor:
    """Return one tensor for the specified class_names, each element as corresponding index in TREE_SPECIES."""
    ll = []
    for c in class_names:
        ll.append(class_to_tensor(c))
    return torch.stack(ll)


def class_to_tensor(class_name: str) -> torch.Tensor:
    """Return TREE_SPECIES index as tensor for the specified class_name, ex. lombardy_poplar -> tensor(5)."""
    return torch.tensor(TREE_SPECIES.index(class_name))


def remove_empty_las_files(start_dir: str, verbose: bool = True) -> None:
    """Remove all files in start_dir that are empty and end with .las."""
    for path, _, files in os.walk(start_dir):
        for f in files:
            full_file_name = os.path.join(path, f)
            # if "000" not in f and f.endswith(".las"):
            #     os.remove(full_file_name)
            #     if verbose:
            #         print(f"Remove {full_file_name} as it it is byproduct of data generation.")
            if f.endswith(".las") and os.stat(full_file_name).st_size == 0:
                os.remove(full_file_name)
                if verbose:
                    print(f"Remove {full_file_name}.")

def print_memory():
    """Print the GPU memory. Use for debugging."""
    total = torch.cuda.get_device_properties(0).total_memory
    reserved = torch.cuda.memory_reserved(0)
    allocated = torch.cuda.memory_allocated(0)
    free = reserved - allocated
    print(f"Memory: Total {total} | Reserved {reserved} | Allocated {allocated} | Free memory {free}")