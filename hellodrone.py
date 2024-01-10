# ready to run example: PythonClient/multirotor/hello_drone.py
# note: async methods take a long time to execute, add join() to wait for it finish 
# NED coordinates: +x:forward +y:right +z:down
# todo:
# 1) set global coordinate, zero origin
import airsim
import os
import time

from controller_m.gen_traj import Generate
lead = "Drone_L"
chase = "Drone_C"

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()

curr_state = client.simGetVehiclePose(lead)
print("lead state", curr_state)
# curr_state.position.z_val = 0
# client.simSetVehiclePose(curr_state, True, lead)
# time.sleep(2)
# curr_state = client.simGetVehiclePose(lead)
# print("lead state", curr_state)

# curr_state2 = client.getMultirotorState(chase)
# print("chaser state", curr_state2)

client.enableApiControl(True,lead)
client.armDisarm(True, lead)
client.takeoffAsync(10.0, lead).join()

client.enableApiControl(True,chase)
client.armDisarm(True, chase)
client.takeoffAsync(10.0, chase).join()

# curr_state = client.getMultirotorState(lead)
# print("lead state", curr_state)
# curr_state2 = client.getMultirotorState(chase)
# print("chaser state", curr_state2)

pose = client.simGetVehiclePose(lead)
print("lead state", pose.position)
####################################################################################################################
cur_x = pose.position.x_val
cur_y = pose.position.y_val
cur_z = pose.position.z_val
start_state = [cur_x,cur_y,cur_z,         0,0,0,   0,0,0]
goal_state =  [cur_x+10,cur_y,cur_z-10,   0,0,1,   0,9.81,0]

d1 = Generate()
time_alloc = 3
position, velocity = d1.generate_traj(starting_state=start_state, goal_state=goal_state, Tf = time_alloc, vis=False)
x = position[:,0]
y = position[:,1]
z = position[:,2]
vx = velocity[:,0]
vy = velocity[:,1]
vz = velocity[:,2]

dt = time_alloc/len(vx)
for i in range(len(vx)):
    print("iter: ",i)
    client.moveByVelocityAsync(vx[i],vy[i],vz[i],dt,vehicle_name=lead).join()
    # time.sleep(2)

# client.moveToZAsync(10,2, vehicle_name=lead).join()
curr_state = client.simGetVehiclePose(lead)
print("lead state", curr_state)

# # Async methods returns Future. Call join() to wait for task to complete.
# client.takeoffAsync().join()
# client.moveToPositionAsync(-10, 10, -10, 5).join()

# # take images
# responses = client.simGetImages([
#     airsim.ImageRequest("0", airsim.ImageType.DepthVis),
#     airsim.ImageRequest("1", airsim.ImageType.DepthPlanar, True)])
# print('Retrieved images: %d', len(responses))

# # do something with the images
# for response in responses:
#     if response.pixels_as_float:
#         print("Type %d, size %d" % (response.image_type, len(response.image_data_float)))
#         airsim.write_pfm(os.path.normpath('py1.pfm'), airsim.get_pfm_array(response))
#     else:
#         print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
#         airsim.write_file(os.path.normpath('py1.png'), response.image_data_uint8)


print("Finished")