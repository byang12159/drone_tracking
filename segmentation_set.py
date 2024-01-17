# # https://github.com/microsoft/AirSim/discussions/4615

# # object ID <-> color: https://microsoft.github.io/AirSim/seg_rgbs.txt

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


# client.simSetSegmentationObjectID("Drone_L", 54, is_name_regex = False)