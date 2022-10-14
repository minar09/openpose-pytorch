import os
import sys
sys.path.insert(0, 'python')
import cv2
import model
import util
import json
from hand import Hand
from body import Body
import matplotlib.pyplot as plt
import copy
import numpy as np

body_estimation = Body('model/body_pose_model.pth')

# image_dir = "D:/Datasets/SeoulTechFashion/same-model-dataset/image/"
# image_dir = "D:/Datasets/viton_resize/test/image/"
image_dir = "/data/matiur/StyleViton/datasets/Fashionade/test_v2/image/"
image_list = os.listdir(image_dir)

for each in image_list:
    test_image = os.path.join(image_dir, each)
    json_file = each[:-4] + "_keypoints.json"
    outjoint_path = os.path.join(image_dir.replace("image/", "pose/"), json_file)
    # outjoint_path = os.path.join(image_dir.replace("image/", "pose-25/"), json_file)

    oriImg = cv2.imread(test_image)  # B,G,R order
    candidate, subset = body_estimation(oriImg)
    canvas = copy.deepcopy(oriImg)
    # print(candidate)
    
    oneperson = { "face_keypoints": [],
                  "pose_keypoints": candidate.flatten().tolist(),
                  "hand_right_keypoints": [],
                  "hand_left_keypoints":[]}

    people   = [oneperson]
    joints_json =  { "version": 1.0, "people": people }
    with open(outjoint_path, 'w') as joint_file:
            json.dump(joints_json, joint_file)
    """canvas = util.draw_bodypose(canvas, candidate, subset)
    plt.imshow(canvas[:, :, [2, 1, 0]])
    plt.axis('off')
    plt.show()"""
