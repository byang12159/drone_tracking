import airsim
import os
import numpy as np
import time

class Perception:
    def __init__(self, client):
        # Set segmentation colors
        # https://github.com/microsoft/AirSim/discussions/4615
        # object ID <-> color: https://microsoft.github.io/AirSim/seg_rgbs.txt
        objects = client.simListSceneObjects()
        # for ob in objects:
        #     print(ob,client.simGetSegmentationObjectID(ob))

        object_name_list = ["road","building","Landscape","Sky","car","building","vegetation","grass","house","tree","Door","wall"]

        print("Reset all object id")
        found = client.simSetSegmentationObjectID("[\w]*", 0, True)
        # print("all object: %r" % (found))
        time.sleep(1)

        for idx,obj_name in enumerate(object_name_list):
            obj_name_reg = r"[\w]" + obj_name + r"[\w]"
        found = client.simSetSegmentationObjectID(obj_name_reg, (idx + 1) % 256, True)
        # print("%s: %r" % (obj_name, found))

        client.simSetSegmentationObjectID("Drone_L", 54, is_name_regex = False)

        
    def capture_RGB(self, client):
        responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
        response = responses[0]
        # get numpy array
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) 
        print(type(img1d))
        print(img1d.shape)
        # reshape array to 3 channel image array H X W X 3
        img_rgb = img1d.reshape(response.height, response.width, 3)
        print(img_rgb.shape)
        # original image is fliped vertically
        # img_rgb = np.flipud(img_rgb)

        # write to png 
        airsim.write_png(os.path.normpath("flight_screenshot" + '.png'), img_rgb) 
        # cv2.imshow("screenshot",response.image_data_uint8)

        return img_rgb
    
    def capture_segment(self, client):
        responses = client.simGetImages([airsim.ImageRequest("5", airsim.ImageType.Scene, False, False)])
        response = responses[0]
        # get numpy array
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) 
        print(type(img1d))
        print(img1d.shape)
        # reshape array to 3 channel image array H X W X 3
        img_rgb = img1d.reshape(response.height, response.width, 3)
        print(img_rgb.shape)
        # original image is fliped vertically
        # img_rgb = np.flipud(img_rgb)

        # write to png 
        airsim.write_png(os.path.normpath("flight_screenshot" + '.png'), img_rgb) 
        # cv2.imshow("screenshot",response.image_data_uint8)

        return img_rgb