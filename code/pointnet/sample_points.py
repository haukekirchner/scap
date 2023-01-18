from utils import PointCloudDataSet, SamplePoints
import laspy
from torchvision import transforms
import numpy as np
from laspy.file import File
import os
import shutil


# setup dir
project_dir = "/scratch/projects/workshops/forest"
data_folder = project_dir + "/synthetic_trees_ten"
new_data_dir = project_dir + "/synthetic_trees_ten_sampled"
train_dir = new_data_dir + "/train"
test_dir = new_data_dir + "/test"

def force_mkdir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)

force_mkdir(new_data_dir)
force_mkdir(train_dir)
force_mkdir(test_dir)

shutil.copyfile(data_folder + "/tree_list.csv",
                new_data_dir + "/tree_list.csv")

#######################################################################
# functions
#######################################################################

def sample_points(data, dir):
    for i in range(len(data.all_files)):
        print(i)
        print(data[1]["label"])
        print(data.all_files[i])
        print(data.all_files[i].split('/')[-1])
        print(data[i]["points"])
        print(type(data[i]["points"]))

        filename = data.all_files[i].split('/')[-1]



        las = laspy.read(data.all_files[i])

        las_new = laspy.create(point_format=las.header.point_format, file_version=las.header.version)

        las_new.x = data[i]["points"][:, 0]
        las_new.y = data[i]["points"][:, 1]
        las_new.z = data[i]["points"][:, 2]

        full_file_name = dir + "/" + filename

        las_new.write(full_file_name)
        print("File saved to : " + full_file_name)




transformations = transforms.Compose(
    [SamplePoints(1024, sample_method="random")])
# It would be also possible to sample the farthest points:
# transformations = transforms.Compose([SamplePoints(1024, sample_method = "farthest_points")])


print(
"########################################################################",
"start sample train data",
"########################################################################")

data = PointCloudDataSet(data_folder, train=True,
                        transform=transformations)

sample_points(data, dir = train_dir)

print(
"########################################################################",
"start sample test data",
"########################################################################")

data = PointCloudDataSet(data_folder, train=False,
                        transform=transformations)

sample_points(data, dir = test_dir)