import os
import numpy as np
import json
import pickle
import os
import matplotlib.pyplot as plt
import cv2

# simulation dataset
home_path = "path_to_dataset/mujoco_dataset/"
child_path = "mujoco_dataset_pick_push_RGBD_different_angles_fast_gripper_224/"
trial = home_path + child_path

mean_joint = np.array(
    [
        -0.24110285,
        0.35597661,
        -1.95778424,
        1.21541062,
        1.36834915,
        -0.0538717,
        0.33596534,
    ]
)
std_joint = np.array(
    [0.62294875, 0.22280216, 0.50348846, 0.95671055, 0.57563566, 0.28734148, 0.33901637]
)

mean_EE = np.array([-0.02838765, 0.20323375, 0.09836011])
std_EE = np.array([0.22767287, 0.10107237, 0.06293893])

mean_obj = np.array([-0.00767178, 0.25526162, 0.06832627])
std_obj = np.array([0.26945344, 0.08801728, 0.05431567])
mean_force = np.array(
    [0.06364701, 0.6060266, 0.2915254, 0.27649295, 0.62418202, 0.28924175]
)
std_force = np.array(
    [0.81733206, 1.46847823, 2.33629577, 1.55631699, 1.49803988, 2.1881589]
)


# create dataset that generate sequential data
def create_pkl(home_path, child_path, trial):
    demo = []
    subdirs = os.listdir(trial)
    for i in subdirs:
        if i != ".DS_Store":
            path = trial + str(i) + "/states.json"
            f = open(path)
            data = json.load(f)
            for idx in range(len(data) - 1):
                item_1 = i
                joint = np.array(data[idx]["q"])
                joint = (joint - mean_joint) / std_joint
                item_2 = joint
                joint = np.array(data[idx + 1]["q"])
                joint = (joint - mean_joint) / std_joint
                item_3 = joint

                x = data[idx]["objects_to_track"]["EE"]["xyz"][0]
                y = data[idx]["objects_to_track"]["EE"]["xyz"][1]
                z = data[idx]["objects_to_track"]["EE"]["xyz"][2]
                EE = np.array([x, y, z])
                EE = (EE - mean_EE) / std_EE
                item_4 = EE

                x = data[idx + 1]["objects_to_track"]["EE"]["xyz"][0]
                y = data[idx + 1]["objects_to_track"]["EE"]["xyz"][1]
                z = data[idx + 1]["objects_to_track"]["EE"]["xyz"][2]
                EE = np.array([x, y, z])
                EE = (EE - mean_EE) / std_EE
                item_5 = EE

                item_6 = child_path + str(i) + "/" + str(idx + 1) + ".png"
                item_7 = child_path + str(i) + "/" + str(idx + 1) + "_depth_map.npy"
                row = [item_1, item_2, item_3, item_4, item_5, item_6, item_7]
                demo.append(row)
        else:
            pass
    with open("UR5_sim_data_train.pkl", "wb") as f:
        pickle.dump(demo, f)


def create_test_pkl(home_path, child_path, trial):
    demo = []
    i = 1602
    path = trial + str(i) + "/states.json"
    f = open(path)
    data = json.load(f)
    for idx in range(len(data) - 1):
        item_1 = i
        joint = np.array(data[idx]["q"])
        joint = (joint - mean_joint) / std_joint
        item_2 = joint
        joint = np.array(data[idx + 1]["q"])
        joint = (joint - mean_joint) / std_joint
        item_3 = joint

        x = data[idx]["objects_to_track"]["EE"]["xyz"][0]
        y = data[idx]["objects_to_track"]["EE"]["xyz"][1]
        z = data[idx]["objects_to_track"]["EE"]["xyz"][2]
        EE = np.array([x, y, z])
        EE = (EE - mean_EE) / std_EE
        item_4 = EE

        x = data[idx + 1]["objects_to_track"]["EE"]["xyz"][0]
        y = data[idx + 1]["objects_to_track"]["EE"]["xyz"][1]
        z = data[idx + 1]["objects_to_track"]["EE"]["xyz"][2]
        EE = np.array([x, y, z])
        EE = (EE - mean_EE) / std_EE
        item_5 = EE

        item_6 = child_path + str(i) + "/" + str(idx + 1) + ".png"
        item_7 = child_path + str(i) + "/" + str(idx + 1) + "_depth_map.npy"
        row = [item_1, item_2, item_3, item_4, item_5, item_6, item_7]
        demo.append(row)
    with open("UR5_sim_data_test_02.pkl", "wb") as f:
        pickle.dump(demo, f)


def create_push_pkl(home_path, child_path, trial):
    demo = []
    subdirs = sorted(os.listdir(trial))
    for i in range(2000):
        if i != ".DS_Store":
            path = trial + str(i) + "/states.json"
            f = open(path)
            data = json.load(f)
            for idx in range(len(data) - 1):
                item_1 = i
                joint = np.array(data[idx]["q"])
                joint = (joint - mean_joint) / std_joint
                item_2 = joint
                joint = np.array(data[idx + 1]["q"])
                joint = (joint - mean_joint) / std_joint
                item_3 = joint

                x = data[idx]["objects_to_track"]["EE"]["xyz"][0]
                y = data[idx]["objects_to_track"]["EE"]["xyz"][1]
                z = data[idx]["objects_to_track"]["EE"]["xyz"][2]
                EE = np.array([x, y, z])
                EE = (EE - mean_EE) / std_EE
                item_4 = EE

                x = data[idx + 1]["objects_to_track"]["EE"]["xyz"][0]
                y = data[idx + 1]["objects_to_track"]["EE"]["xyz"][1]
                z = data[idx + 1]["objects_to_track"]["EE"]["xyz"][2]
                EE = np.array([x, y, z])
                EE = (EE - mean_EE) / std_EE
                item_5 = EE

                x = data[idx + 1]["objects_to_track"][data[idx + 1]["goal_object"]][
                    "xyz"
                ][0]
                y = data[idx + 1]["objects_to_track"][data[idx + 1]["goal_object"]][
                    "xyz"
                ][1]
                z = data[idx + 1]["objects_to_track"][data[idx + 1]["goal_object"]][
                    "xyz"
                ][2]
                obj = np.array([x, y, z])
                obj = (obj - mean_obj) / std_obj
                item_8 = obj

                force = np.array(data[idx + 1]["touch_sensor"])
                force = (force - mean_force) / std_force

                item_6 = child_path + str(i) + "/" + str(idx + 1) + ".png"
                item_7 = child_path + str(i) + "/" + str(idx + 1) + "_depth_map.npy"
                row = [
                    item_1,
                    item_2,
                    item_3,
                    item_4,
                    item_5,
                    item_6,
                    item_7,
                    item_8,
                    force,
                ]
                demo.append(row)
        else:
            pass
    with open("UR5_push_data_train.pkl", "wb") as f:
        pickle.dump(demo, f)


def create_push_test_pkl(home_path, child_path, trial):
    demo = []
    subdirs = sorted(os.listdir(trial))
    i = 2002
    path = trial + str(i) + "/states.json"
    f = open(path)
    data = json.load(f)
    for idx in range(len(data) - 1):
        item_1 = i
        joint = np.array(data[idx]["q"])
        joint = (joint - mean_joint) / std_joint
        item_2 = joint
        joint = np.array(data[idx + 1]["q"])
        joint = (joint - mean_joint) / std_joint
        item_3 = joint

        x = data[idx]["objects_to_track"]["EE"]["xyz"][0]
        y = data[idx]["objects_to_track"]["EE"]["xyz"][1]
        z = data[idx]["objects_to_track"]["EE"]["xyz"][2]
        EE = np.array([x, y, z])
        EE = (EE - mean_EE) / std_EE
        item_4 = EE

        x = data[idx + 1]["objects_to_track"]["EE"]["xyz"][0]
        y = data[idx + 1]["objects_to_track"]["EE"]["xyz"][1]
        z = data[idx + 1]["objects_to_track"]["EE"]["xyz"][2]
        EE = np.array([x, y, z])
        EE = (EE - mean_EE) / std_EE
        item_5 = EE

        x = data[idx + 1]["objects_to_track"][data[idx + 1]["goal_object"]]["xyz"][0]
        y = data[idx + 1]["objects_to_track"][data[idx + 1]["goal_object"]]["xyz"][1]
        z = data[idx + 1]["objects_to_track"][data[idx + 1]["goal_object"]]["xyz"][2]
        obj = np.array([x, y, z])
        obj = (obj - mean_obj) / std_obj
        item_8 = obj

        force = np.array(data[idx + 1]["touch_sensor"])
        force = (force - mean_force) / std_force

        item_6 = child_path + str(i) + "/" + str(idx + 1) + ".png"
        item_7 = child_path + str(i) + "/" + str(idx + 1) + "_depth_map.npy"
        row = [item_1, item_2, item_3, item_4, item_5, item_6, item_7, item_8, force]
        demo.append(row)
    with open("UR5_push_data_test.pkl", "wb") as f:
        pickle.dump(demo, f)


# create real seq data
def create_real_pkl(home_path):
    demo = []
    # the real dataset 1
    child_path = "data_real_matched_q/"
    trial = home_path + child_path
    for i in range(29):
        if i != 6:
            path = trial + str(i) + "/states.json"
            f = open(path)
            data = json.load(f)
            for idx in range(len(data) - 2):
                item_1 = child_path + str(i)
                joint = np.array(data[idx]["q"])
                joint = (joint - mean_joint) / std_joint
                item_2 = joint
                joint = np.array(data[idx + 1]["q"])
                joint = (joint - mean_joint) / std_joint
                item_3 = joint
                x = data[idx]["objects_to_track"]["EE"]["xyz"][0]
                y = data[idx]["objects_to_track"]["EE"]["xyz"][1]
                z = data[idx]["objects_to_track"]["EE"]["xyz"][2]
                EE = np.array([x, y, z])
                EE = (EE - mean_EE) / std_EE
                item_4 = EE
                x = data[idx + 1]["objects_to_track"]["EE"]["xyz"][0]
                y = data[idx + 1]["objects_to_track"]["EE"]["xyz"][1]
                z = data[idx + 1]["objects_to_track"]["EE"]["xyz"][2]
                EE = np.array([x, y, z])
                EE = (EE - mean_EE) / std_EE
                item_5 = EE
                item_6 = (
                    child_path + str(i) + "/real_" + str(idx + 1) + "_processed.png"
                )
                row = [item_1, item_2, item_3, item_4, item_5, item_6]
                demo.append(row)

    # the real dataset 2
    child_path = "data_real_matched_q_grid/"
    trial = home_path + child_path
    for i in range(49):
        path = trial + str(i) + "/states.json"
        f = open(path)
        data = json.load(f)
        for idx in range(len(data) - 2):
            item_1 = child_path + str(i)
            joint = np.array(data[idx]["q"])
            joint = (joint - mean_joint) / std_joint
            item_2 = joint
            joint = np.array(data[idx + 1]["q"])
            joint = (joint - mean_joint) / std_joint
            item_3 = joint
            x = data[idx]["objects_to_track"]["EE"]["xyz"][0]
            y = data[idx]["objects_to_track"]["EE"]["xyz"][1]
            z = data[idx]["objects_to_track"]["EE"]["xyz"][2]
            EE = np.array([x, y, z])
            EE = (EE - mean_EE) / std_EE
            item_4 = EE
            x = data[idx + 1]["objects_to_track"]["EE"]["xyz"][0]
            y = data[idx + 1]["objects_to_track"]["EE"]["xyz"][1]
            z = data[idx + 1]["objects_to_track"]["EE"]["xyz"][2]
            EE = np.array([x, y, z])
            EE = (EE - mean_EE) / std_EE
            item_5 = EE
            item_6 = child_path + str(i) + "/real_" + str(idx + 1) + "_processed.png"
            row = [item_1, item_2, item_3, item_4, item_5, item_6]
            demo.append(row)

    # the real dataset 3
    child_path = "data_real_matched_q_grid_2/"
    trial = home_path + child_path
    for i in range(50, 79):
        path = trial + str(i) + "/states.json"
        f = open(path)
        data = json.load(f)
        for idx in range(len(data) - 2):
            item_1 = child_path + str(i)
            joint = np.array(data[idx]["q"])
            joint = (joint - mean_joint) / std_joint
            item_2 = joint
            joint = np.array(data[idx + 1]["q"])
            joint = (joint - mean_joint) / std_joint
            item_3 = joint
            x = data[idx]["objects_to_track"]["EE"]["xyz"][0]
            y = data[idx]["objects_to_track"]["EE"]["xyz"][1]
            z = data[idx]["objects_to_track"]["EE"]["xyz"][2]
            EE = np.array([x, y, z])
            EE = (EE - mean_EE) / std_EE
            item_4 = EE
            x = data[idx + 1]["objects_to_track"]["EE"]["xyz"][0]
            y = data[idx + 1]["objects_to_track"]["EE"]["xyz"][1]
            z = data[idx + 1]["objects_to_track"]["EE"]["xyz"][2]
            EE = np.array([x, y, z])
            EE = (EE - mean_EE) / std_EE
            item_5 = EE
            item_6 = child_path + str(i) + "/real_" + str(idx + 1) + "_processed.png"
            row = [item_1, item_2, item_3, item_4, item_5, item_6]
            demo.append(row)
    with open("real_train_data.pkl", "wb") as f:
        pickle.dump(demo, f)


def create_real_test_pkl(home_path):
    demo = []
    i = 30

    # the real dataset 1
    child_path = "data_real_matched_q/"
    trial = home_path + child_path
    path = trial + str(i) + "/states.json"
    f = open(path)
    data = json.load(f)
    for idx in range(len(data) - 2):
        item_1 = child_path + str(i)
        joint = np.array(data[idx]["q"])
        joint = (joint - mean_joint) / std_joint
        item_2 = joint
        joint = np.array(data[idx + 1]["q"])
        joint = (joint - mean_joint) / std_joint
        item_3 = joint
        x = data[idx]["objects_to_track"]["EE"]["xyz"][0]
        y = data[idx]["objects_to_track"]["EE"]["xyz"][1]
        z = data[idx]["objects_to_track"]["EE"]["xyz"][2]
        EE = np.array([x, y, z])
        EE = (EE - mean_EE) / std_EE
        item_4 = EE
        x = data[idx + 1]["objects_to_track"]["EE"]["xyz"][0]
        y = data[idx + 1]["objects_to_track"]["EE"]["xyz"][1]
        z = data[idx + 1]["objects_to_track"]["EE"]["xyz"][2]
        EE = np.array([x, y, z])
        EE = (EE - mean_EE) / std_EE
        item_5 = EE
        item_6 = child_path + str(i) + "/real_" + str(idx + 1) + "_processed.png"
        row = [item_1, item_2, item_3, item_4, item_5, item_6]
        demo.append(row)
    with open("real_test_data.pkl", "wb") as f:
        pickle.dump(demo, f)


def main():
    ################## simulation dataset for manipulation #################
    home_path = "path_t0_dataset/mujoco_dataset/"
    child_path = "mujoco_dataset_pick_push_RGBD_different_angles_fast_gripper_224/"
    trial = home_path + child_path
    create_pkl(home_path, child_path, trial)

    child_path = "mujoco_dataset_pick_push_RGBD_different_angles_fast_gripper_224_test/"
    trial = home_path + child_path
    create_test_pkl(home_path, child_path, trial)

    data_tmp = pickle.load(open("UR5_sim_data_test_02.pkl", "rb"))
    print(data_tmp[0])

    # ################# simulation dataset for push #################
    # home_path = "/tf/datasets/"
    # child_path = "mujoco_ur5_touch_sensor/"

    # trial = home_path + child_path
    # create_push_pkl(home_path, child_path, trial)
    # create_push_test_pkl(home_path, child_path, trial)

    # data_tmp = pickle.load(open("UR5_push_data_train.pkl", "rb"))
    # # obj_loc = []
    # # force = []
    # # for i in range(len(data_tmp)):
    # #     obj_loc.append(data_tmp[i][-2])
    # #     force.append(data_tmp[i][-1])
    # # obj_loc = np.array(obj_loc)
    # # force = np.array(force)
    # # print(np.mean(obj_loc, axis=0))
    # # print(np.std(obj_loc, axis=0))
    # # print(np.mean(force, axis=0))
    # # print(np.std(force, axis=0))

    ################# real dataset #################
    # home_path = 'path_to_dataset/real_dataset/real_data/'
    # create_real_pkl(home_path)
    # create_real_test_pkl(home_path)

    # data_tmp = pickle.load(open("real_test_data.pkl", "rb"))
    # print(data_tmp[0][10])


if __name__ == "__main__":
    main()
