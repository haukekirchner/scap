from utils import PointCloudDataSet, SamplePoints
import laspy
from torchvision import transforms
import numpy as np
from laspy.file import File


batch_size = 1
data_folder = "/scratch/projects/workshops/forest/synthetic_trees_ten"

transformations = transforms.Compose(
    [SamplePoints(1024, sample_method="random")])
# It would be also possible to sample the farthest points:
# transformations = transforms.Compose([SamplePoints(1024, sample_method = "farthest_points")])

data = PointCloudDataSet(data_folder, train=True,
                        transform=transformations)

for i in range(len(data.all_files)):
    print(i)
    print(data[1]["label"])
    print(data.all_files[i])
    print(data.all_files[i].split('/')[-1])
    print(data[i]["points"])
    print(type(data[i]["points"]))
    print("---------------------")

    inFile = File(data.all_files[i], mode = "r")

    outFile = File("~/pc.las", mode = "w",
                   header = inFile.header)
    outFile.points = points_kept
    outFile.close()
