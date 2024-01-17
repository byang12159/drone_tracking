import airsim
import os
import numpy as np
import time
import cv2

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
        # object 54: RBG [120, 0, 200]

        
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
    
    def capture_segment(self, client,set_compress=False):
        # get segmentation image in various formats
        responses = client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Segmentation, True), #depth in perspective projection
            airsim.ImageRequest("0", airsim.ImageType.Segmentation, pixels_as_float=False, compress=set_compress)])  #scene vision image in uncompressed RGBA array
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

                #find unique colors
                # print(np.unique(img_rgb[:,:,0], return_counts=True)) #red
                # print(np.unique(img_rgb[:,:,1], return_counts=True)) #green
                # print(np.unique(img_rgb[:,:,2], return_counts=True)) #blue  
        return img_rgb
    
    def segment_detect(self, img):
        # Detect lead drone in segmented image using traditional CV techniques

        #check img color and apply fitler
        print(img)
        pass

    # Read the image
    # image = cv2.imread('your_image_path.jpg')  # Replace with the path to your image

    # # Convert the image to the HSV color space
    # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # # Define the lower and upper bounds for the red color in HSV
    # lower_red = np.array([0, 100, 100])
    # upper_red = np.array([10, 255, 255])

    # # Create a binary mask for the red color
    # mask = cv2.inRange(hsv, lower_red, upper_red)

    # # Find contours in the binary mask
    # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # # Iterate over the contours and find the bounding box
    # for contour in contours:
    #     # Ignore small contours
    #     if cv2.contourArea(contour) > 100:
    #         # Approximate the contour to a polygon
    #         epsilon = 0.02 * cv2.arcLength(contour, True)
    #         approx = cv2.approxPolyDP(contour, epsilon, True)

    #         # Draw the bounding box around the object
    #         cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)

    #         # Draw circles at the corners of the bounding box
    #         for point in approx:
    #             cv2.circle(image, tuple(point[0]), 5, (255, 0, 0), -1)

    # # Display the result
    # cv2.imshow('Bounding Box', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
