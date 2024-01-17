# import airsim
# import time
# client = airsim.VehicleClient()
# client.confirmConnection()

# objects = client.simListSceneObjects()
# for ob in objects:
#     print(ob,client.simGetSegmentationObjectID(ob))

# object_name_list = ["road","building","Landscape","Sky","car","building","vegetation","grass","house","tree","Door","wall"]

# print("Reset all object id")
# found = client.simSetSegmentationObjectID("[\w]*", 0, True)
# print("all object: %r" % (found))
# time.sleep(1)

# for idx,obj_name in enumerate(object_name_list):
#     obj_name_reg = r"[\w]" + obj_name + r"[\w]"
# found = client.simSetSegmentationObjectID(obj_name_reg, (idx + 1) % 256, True)
# print("%s: %r" % (obj_name, found))


import airsim
import os
import time
import numpy as np
import cv2


from controller_m.gen_traj import Generate
from perception.perception import Perception
lead = "Drone_L"
chase = "Drone_C"

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
print("Finished")


# client.simAddDetectionFilterMeshName(camera_name = "Camera_main",image_type =  "0",mesh_name="s*", vehicle_name = chase)
# success = client.simSetSegmentationObjectID("Ground", 22)
# print(success)

#get segmentation image in various formats
responses = client.simGetImages([
    airsim.ImageRequest("0", airsim.ImageType.Segmentation, True), #depth in perspective projection
    airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, True)])  #scene vision image in uncompressed RGBA array
print('Retrieved images: %d', len(responses))

#save segmentation images in various formats
for idx, response in enumerate(responses):
    filename = 'c:/temp/py_seg_' + str(idx)

    if response.pixels_as_float:
        print("Type %d, size %d" % (response.image_type, len(response.image_data_float)))
        airsim.write_pfm(os.path.normpath("segment1"+ '.pfm'), airsim.get_pfm_array(response))
    elif response.compress: #png format
        print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
        airsim.write_file(os.path.normpath("segment22" + '.png'), response.image_data_uint8)
    else: #uncompressed array - numpy demo
        print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) #get numpy array
        img_rgb = img1d.reshape(response.height, response.width, 3) #reshape array to 3 channel image array H X W X 3
        cv2.imwrite(os.path.normpath("segment3"+ '.png'), img_rgb) # write to png

        # #find unique colors
        # print(np.unique(img_rgb[:,:,0], return_counts=True)) #red
        # print(np.unique(img_rgb[:,:,1], return_counts=True)) #green
        # print(np.unique(img_rgb[:,:,2], return_counts=True)) #blue  
return img_rgb