import sys
import cv2 as cv
import numpy as np
import os
import math
from streamprofile_batch import StreamProfileRequests
from matplotlib import pyplot as plt
import time

# BRISK Constants
BRISK_LOWE_DIST_RATIO = .75
BRISK_MIN_MATCH_COUNT = 4

# Intinsic Camera Matrix
CAM_INTR = np.array([[1489.6, 0, 920.5],[0, 1476.1, 571.6],[0, 0, 1]])

# Camera object
user = None #CHANGE
password = None
profile = None
host = None
url = f'rtsp://{user}:{password}@{host}/axis-media/media.amp?streamprofile={profile}'
cam = StreamProfileRequests()
cam.set_credentials(host, user, password)

def save_picture(frm):
    '''
    Saves view of camera as jpg image
    :param frm: frame to save
    :return: (string) filepath of frame saved
    '''
    # Display the resulting frame
    # cv.imshow('Frame',frm)
    position=cam.get_position()
    pan=position["pan"]
    tilt=position["tilt"]
    foldername = 'cam2/' #CHANGE: folder to save images taken
    filename=f'{host}_p{pan}_t{tilt}.jpg'
    filepath = foldername+filename
    print("writing out file to", filepath)
    cv.imwrite(filepath,frm)

    return filepath


def take_picture():
    '''
    Saves view of camera as jpg image
    :return: (string) filename of frame saved
    '''
    print("opening video")
    cap = cv.VideoCapture(url,cv.CAP_FFMPEG)

    if not cap.isOpened():
        print('Unable to open stream')
        exit()
        
    sumFrame = None
    sumCnt = 0
    sumSize = 60
    picNum=0

    ret,frame = cap.read()

    print("Taking a picture at position: ", cam.get_position()["pan"], cam.get_position()["tilt"])
    for i in range(1): 
        while ret:
            if sumFrame is None:
                sumFrame = frame.astype(np.float64)
                sumCnt = 0
            else:
                sumFrame = sumFrame + frame
                sumCnt +=1
            if sumCnt >= sumSize:
                sumFrame = sumFrame / sumSize
                frm = sumFrame.astype(np.uint8)
                if picNum==1:
                    # Saving every other picture
                    filename = save_picture(frm)
                    sumFrame = None
                    picNum=0
                    break
                else:
                    picNum=picNum+1
                    sumFrame = None

            ret, frame = cap.read()

    # When everything done, release the video capture object 
    cap.release()
    
    # Closes all the frames
    cv.destroyAllWindows()    

    return filename

def read_image(image_path):
    '''
    Read in image file
    :param image_path: directory path of image file
    :return: (np array) image
    '''
    print(": Reading image: ", image_path)
    img = cv.imread(image_path, cv.COLOR_BGR2GRAY)
    return img


def read_snapshot_360(foldername):
    '''
    NOT USED
    Reads in 360 degrees of images at 20 degree intervals starting from position of image
    :param foldername: name of folder where images are
    :return: array of images
    '''
    images=[]
    count=0
    for filename in os.listdir(foldername):
        if filename[-3:]=="jpg" :
            images.append(filename)
            count+=1
    print(f'loaded {count} images from {foldername}')
    return images

def calculate_rot_matrix(ref_img, img):
    '''
    Calculates homography and rotation matrix of an image to a reference image with BRISK feature detection
    :param ref_img: reference image
    :param img: destination image
    :return: (int, np array) number of good matches, rotation matrix from homography
    '''
    # Initiate BRISK detector
    brisk = cv.BRISK_create()

    # Find the keypoints and descriptors with BRISK
    kp_rimg, descr_rimg = brisk.detectAndCompute(ref_img, None)
    kp_img, descr_img = brisk.detectAndCompute(img, None)

    # Matching with FLANN
    FLANN_INDEX_LSH = 6
    flann_params= dict(algorithm = FLANN_INDEX_LSH,
                                table_number = 6, # 12
                                key_size = 12,     # 20
                                multi_probe_level = 1) #2
    matcher = cv.FlannBasedMatcher(flann_params, {})
    matches = matcher.knnMatch(descr_rimg, descr_img, k = 2) 
    # Store all the good matches as per Lowe's ratio test
    good = []
    for m,n in matches:
        if m.distance < BRISK_LOWE_DIST_RATIO*n.distance:
            good.append(m)

    # Found enough matches, success
    if len(good)>BRISK_MIN_MATCH_COUNT:
        # print("Success!" ) 
        dst_pts = np.float32([ kp_rimg[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        src_pts = np.float32([ kp_img[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    else:
        raise Exception("Not enough matches are found - {}/{}".format(len(good), BRISK_MIN_MATCH_COUNT)) 
   
    # Calculate homography
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
    M_ret, M_rot, M_tran, M_norm = cv.decomposeHomographyMat(M, CAM_INTR)

    return len(good), M_rot

def isRotationMatrix(R) :
    '''
    Checks rotation matrix validity
    :param R: rotation matrix
    :return: (bool) validity of rotation matrix
    '''
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
 
def rotation_matrix_to_euler(R) :
    '''
    Derives Euler angles from rotation matrix
    The result is the same as MATLAB except the order of the euler angles ( x and z are swapped ).
    :param R: rotation matrix
    :return: (np array) tuple of Euler angles
    '''
    if not isRotationMatrix(R):
        print("ERROR IN ROT MATRIX")
        return [0,0,0]
 
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])

def calculate_degrees(rot):
    '''
    Calculates angles in degrees from rotation matrix
    :param rot: rotation matrix
    :return: (np array) tuple of angles in degrees
    '''
    euler_angles = rotation_matrix_to_euler(rot)
    degrees_angles = euler_angles
    for j in range(3):
        degrees_angles[j]=degrees_angles[j]*180/3.1415
    return degrees_angles

def check_rot_array_pan(rot_degrees1, rot_degrees2):
    '''
    Sanity check for rotation array(s) calculated for panning
    :param rot_degrees1: array of rotation in degrees
    :param rot_degrees2: array of rotation in degrees
    :return: None
    '''
    # Checks if all degrees<100, otherwise images are of completely different views and should not have found good feature points
    c1 = all(i < 100 for i in rot_degrees1)
    # Checks if panning degrees are of the same sign
    c3 = rot_degrees1[1]<=0 and rot_degrees1[2]<=0 or rot_degrees2[1]>=0 and rot_degrees2[2]>=0

    # If second rotation matrix exists
    if rot_degrees2 is not None:
        c2 = all(i < 100 for i in rot_degrees2)
        c4 =  rot_degrees2[1]<=0 and rot_degrees2[2]<=0 or rot_degrees2[1]>=0 and rot_degrees2[2]>=0
        assert (c1 and c2)
        # Sign uniformity is not applicable for small angles
        if not (abs(rot_degrees1[1])<1 and abs(rot_degrees1[2])<1 and abs(rot_degrees2[1])<1 and abs(rot_degrees2[2])<1):
            assert(c3 or c4)
    # If only one rotation matrix exists
    else:
        assert(c1)
        if not (abs(rot_degrees1[1])<1 and abs(rot_degrees1[2])<1):
            assert(c3)  

def check_rot_array_tilt(rot_degrees1, rot_degrees2):
    '''
    Sanity check rotation array(s) for tilting
    :param rot_degrees1: array of rotation in degrees
    :param rot_degrees2: array of rotation in degrees
    :return: None
    '''
    # Checks if panning angles<2, should have been corrected by panning already
    c1 = abs(rot_degrees1[1])<2 and abs(rot_degrees1[2])<2
    if rot_degrees2 is not None:
        c2 = abs(rot_degrees2[1])<2 and abs(rot_degrees2[2])<2
        assert(c1 and c2)
    else:
        assert(c1)


def main():
    found_ref_view = False # Found enough matching feature points
    reached_edge = False # Camera is pointing down completely at the edge of tilt boundary

    ref_image_filepath = "cam2/ref_image.jpg" #CHANGE: reference image filepath

    ref_image = read_image(ref_image_filepath)

    tilted = 0
    panned = 0

    while not found_ref_view:
        while panned <= 18:
            print(f"[SCANNING] Panning #{panned} ...")
            snapshot_filepath = take_picture()
            snap_image = read_image(snapshot_filepath)
            num_points_matched, rot_matrices = calculate_rot_matrix(ref_img=ref_image, img=snap_image)
            if num_points_matched>200:
                print(f"[SCANNING] Found a good match with {num_points_matched} good points")
                # plt.imshow(image, 'gray'),plt.show()
                # time.sleep(5)
                found_ref_view = True
                recognized_image = snapshot_filepath
                break
            else:
                print(f"[SCANNING] Not enough good matches: {num_points_matched}")
            cam.move(float(cam.get_position()["pan"])+20, float(cam.get_position()["tilt"]))
            panned+=1

        if not found_ref_view:
            # scanned 360 degrees at current tilt angle, tilt down (or up) and pan again
            tilted+=1
            panned = 0
            if not reached_edge:
                print(f"[SCANNING] Tilting #{tilted} ...")
                prev_tilt = float(cam.get_position()["tilt"])
                cam.move(float(cam.get_position()["pan"]), float(cam.get_position()["tilt"])-20)
                cur_tilt = float(cam.get_position()["tilt"])
                if prev_tilt<cur_tilt: #reached the bottom boundary, tilt up instead
                    print("[SCANNING] Reached the bottom")
                    cam.move(cam.get_position()["pan"], cam.get_position()["tilt"]+40)
                    reached_edge = True
            else:
                cam.move(cam.get_position()["pan"], cam.get_position()["tilt"]+20)
                if(tilted==9):
                    # panned 18 * 20 = 360 degrees and tilted 9 * 20 = 180 degrees
                    raise Exception("[SCANNING] Scanned everything, could not find matching snapshot")

    recognized_image_split = recognized_image.split('_') # Parsing saved file's name
    abs_pan = float(recognized_image_split[1][1:])
    abs_tilt = float(recognized_image_split[2][1:][:-4])
    print(f"[SCANNING] Moving to recognized position: pan {abs_pan} tilt {abs_tilt}")

    rot2 = None
    rot_degrees2 = None
    print(f"[CORRECTING] Start panning...")
    rot1 = rot_matrices[0]
    rot_degrees1 = calculate_degrees(rot1)
    if (len(rot_matrices)>2): # 1 or 2 rotation array(s) calculated
        rot2 = rot_matrices[2]
        rot_degrees2 = calculate_degrees(rot2)
        max_degree_pan = max(abs(rot_degrees1[1]), abs(rot_degrees1[2]), abs(rot_degrees2[1]), abs(rot_degrees2[2]))
    else:
        max_degree_pan = max(abs(rot_degrees1[1]), abs(rot_degrees1[2]))
    check_rot_array_pan(rot_degrees1=rot_degrees1, rot_degrees2=rot_degrees2)
    if(rot_degrees1[1]<0):
        max_degree_pan = -max_degree_pan
    
    prev_max_degree_pan = 360
    while(abs(max_degree_pan)>1):
        assert(abs(prev_max_degree_pan)>abs(max_degree_pan)) # Checks that angle difference is getting smaller
        prev_max_degree_pan = max_degree_pan

        print(f"[CORRECTING] Panning by {max_degree_pan}...")
        cam.move(float(cam.get_position()["pan"])-max_degree_pan, float(cam.get_position()["tilt"]))
        new_pan = float(cam.get_position()["pan"])
        new_tilt = float(cam.get_position()["tilt"])
        print(f'[CORRECTING] New position after pan: pan {new_pan}, tilt {new_tilt}')
        new_filepath = take_picture()
        image_moved = read_image(new_filepath)
        num_points_matched, rot_matrices = calculate_rot_matrix(ref_img=ref_image, img=image_moved)

        rot2 = None
        rot_degrees2 = None
        rot1 = rot_matrices[0]
        rot_degrees1 = calculate_degrees(rot1)
        if (len(rot_matrices)>2): # 1 or 2 rotation array(s) calculated
            rot2 = rot_matrices[2]
            rot_degrees2 = calculate_degrees(rot2)
            max_degree_pan = max(abs(rot_degrees1[1]), abs(rot_degrees1[2]), abs(rot_degrees2[1]), abs(rot_degrees2[2]))
        else:
            max_degree_pan = max(abs(rot_degrees1[1]), abs(rot_degrees1[2]))
        if(rot_degrees1[1]<0): # Restoring sign
            max_degree_pan = -max_degree_pan
        check_rot_array_pan(rot_degrees1=rot_degrees1, rot_degrees2=rot_degrees2)
    
    final_pan = float(cam.get_position()["pan"])
    final_tilt = float(cam.get_position()["tilt"])
    print(f"[CORRECTING] Done panning. Position is now pan: {final_pan}, tilt: {final_tilt}")


    print(f"[CORRECTING] Start tilting...")
    if (len(rot_matrices)>2): # 1 or 2 rotation array(s) calculated
        max_degree_tilt = max(abs(rot_degrees1[0]), abs(rot_degrees2[0]))
    else:
        max_degree_tilt = max(abs(rot_degrees1[0]))
    check_rot_array_tilt(rot_degrees1=rot_degrees1, rot_degrees2=rot_degrees2)
    if(rot_degrees1[0]<0): # Restoring sign
        max_degree_tilt = -max_degree_tilt

    prev_max_degree_tilt = 360
    while(abs(max_degree_tilt)>.5):
        assert(abs(prev_max_degree_tilt)>abs(max_degree_tilt)) # Checks that angle difference is getting smaller
        prev_max_degree_tilt = max_degree_tilt

        print(f"[CORRECTING] Tilting by {max_degree_tilt}")
        cam.move(float(cam.get_position()["pan"]), float(cam.get_position()["tilt"])-max_degree_tilt)
        new_pan = float(cam.get_position()["pan"])
        new_tilt = float(cam.get_position()["tilt"])
        print(f'[CORRECTING] New position after tilt: pan: {new_pan}, tilt: {new_tilt}')
        new_filepath = take_picture()
        image_moved = read_image(new_filepath)
        num_points_matched, rot_matrices = calculate_rot_matrix(ref_img=ref_image, img=image_moved)
        
        rot2 = None
        rot_degrees2 = None
        rot1 = rot_matrices[0]
        rot_degrees1 = calculate_degrees(rot1)
        if (len(rot_matrices)>2): # 1 or 2 rotation array(s) calculated
            rot2 = rot_matrices[2]
            rot_degrees2 = calculate_degrees(rot2)
            max_degree_tilt = max(abs(rot_degrees1[0]), abs(rot_degrees2[0]))
        else:
            max_degree_tilt = max(abs(rot_degrees1[0]))
        check_rot_array_tilt(rot_degrees1=rot_degrees1, rot_degrees2=rot_degrees2)
        if(rot_degrees1[0]<0):
            max_degree_tilt = -max_degree_tilt

    print(f'[CORRECTING] Done tilting: position is pan {new_pan}, tilt {new_tilt}')
    print("FINISHED")

if __name__ == '__main__':
    sys.exit(main())