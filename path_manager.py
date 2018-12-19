import os
from time import gmtime, strftime

Project_path = os.getcwd()

Export_path = Project_path + "/export"

Current_time = strftime("%Y%m%d%H%M%S", gmtime())

Step_path = Export_path + "/" + Current_time

Model_path = Step_path + "/model"

Tensorboard_path = Step_path + "/tensorboard"

Summary_path = Step_path + "/summary"

path_arr = [Project_path, Export_path, Step_path,
            Model_path, Tensorboard_path, Summary_path]

for path in path_arr:
    try:
        if not(os.path.isdir(path)):
            os.makedirs(os.path.join(path))
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Failed to create directory!!!!!")
            raise

