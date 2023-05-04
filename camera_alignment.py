import sys
import cv2 as cv
import numpy as np
import os
import math
from streamprofile_batch import StreamProfileRequests
import time
from matplotlib import pyplot as plt

# BRISK Constants
BRISK_LOWE_DIST_RATIO = .75
BRISK_MIN_MATCH_COUNT = 4

# Intinsic Camera Matrix
CAM_INTR = np.array([[1489.6, 0, 920.5],[0, 1476.1, 571.6],[0, 0, 1]])

user = 'root'
password  = 'i24camadmin1025'
profile = '1080p_h264'
host = '10.80.134.62'
url = f'rtsp://{user}:{password}@{host}/axis-media/media.amp?streamprofile={profile}'
cam=StreamProfileRequests()
cam.set_credentials(host, user, password)

def process(frm):
    
    # Display the resulting frame
    # cv.imshow('Frame',frm)
    position=cam.get_position()
    pan=position["pan"]
    tilt=position["tilt"]
    foldername = 'cam2/'
    filename=f'{host}_p{pan}_t{tilt}.jpg'
    print("writing out file to", foldername+filename)
    cv.imwrite(foldername+filename,frm)

    return filename


def take_picture():
    # print("opening video")
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
                # print('new frame')
            else:
                sumFrame = sumFrame + frame
                sumCnt +=1
                # print('add frame')

            if sumCnt >= sumSize:
                sumFrame = sumFrame / sumSize
                
                frm = sumFrame.astype(np.uint8)

                if picNum==1:
                    # print("keeping pic")
                    filename = process(frm)
                    tilt=(float)(cam.get_position()["tilt"])
                    pan=(float)(cam.get_position()["pan"])
                    sumFrame = None
                    picNum=0
                    break
                else:
                    # print("throwing away pic")
                    picNum=picNum+1
                    sumFrame = None

            ret, frame = cap.read()

    # When everything done, release the video capture object 
    cap.release()
    
    # Closes all the frames
    cv.destroyAllWindows()    

    return filename

def read_image(foldername, image_filename):
    '''
    Read in image file
    :param foldername: name of folder where image is
    :param image_filename: name of image file
    :return: image as np array
    '''
    print(": Reading image: ", foldername+image_filename)
    img = cv.imread(foldername+image_filename, cv.COLOR_BGR2GRAY)
    return img


def read_snapshot_360(foldername):
    '''
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
    Calculates rotation for an image with BRISK from homography
    :param ref_img: reference image
    :param img: destination image
    :return: number of good matches, rotation matrix from homography
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
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
 
def rotationMatrixToEulerAngles(R) :
    '''
    Derives Euler angles from rotation matrix
    The result is the same as MATLAB except the order of the euler angles ( x and z are swapped ).
    :param R: rotation matrix
    :return: tuple of Euler angles
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
    Calculates array of angles in degrees from rotation matrix
    :param rot: rotation matrix
    :return: tuple of angles in degrees
    '''
    euler_angles = rotationMatrixToEulerAngles(rot)
    degrees_angles = euler_angles
    for j in range(3):
        degrees_angles[j]=degrees_angles[j]*180/3.1415
    return degrees_angles


# sanity check for panning
def check_rot_array_pan(rot_degrees1, rot_degrees2):
    '''
    Sanity check for panning
    :param rot_degrees1: array of rotation in degrees
    :param rot_degrees2: array of rotation in degrees
    :return: None
    '''
    c1 = rot_degrees1[0]<rot_degrees1[1] and rot_degrees1[0]<rot_degrees1[2]
    c3 = rot_degrees1[0]>rot_degrees1[1] and rot_degrees1[0]>rot_degrees1[2]
    c5 = all(i < 100 for i in rot_degrees1)
    c7 = rot_degrees1[1]<=0 and rot_degrees1[2]<=0 or rot_degrees2[1]>=0 and rot_degrees2[2]>=0
    if rot_degrees2 is not None:
        c2 = rot_degrees2[0]<rot_degrees2[1] and rot_degrees2[0]<rot_degrees2[2]
        c4 = rot_degrees2[0]>rot_degrees2[1] and rot_degrees2[0]>rot_degrees2[2]
        c6 = all(i < 100 for i in rot_degrees2)
        c8 =  rot_degrees2[1]<=0 and rot_degrees2[2]<=0 or rot_degrees2[1]>=0 and rot_degrees2[2]>=0
        # print("checking degrees are of the similar relative magnitude...")
        # assert((c1 and c2) or (c3 and c4))
        # print("checking degrees are less than 100...")
        assert (c5 and c6)
        # print("checking degrees are of the same sign...")
        # small enough
        if not (abs(rot_degrees1[1])<1 and abs(rot_degrees1[2])<1 and abs(rot_degrees2[1])<1 and abs(rot_degrees2[2])<1):
            assert(c7 or c8)

    else:
        # print("checking degrees are of the similar relative magnitude...")
        # assert(c1 or c3)
        # print("checking degrees are less than 100...")
        assert(c5)
        # print("checking degrees are of the same sign...")
        if not (abs(rot_degrees1[1])<1 and abs(rot_degrees1[2])<1):
            assert(c7)  

def check_rot_arr_tilt(rot_degrees1, rot_degrees2):
    '''
    Sanity check for tilting
    :param rot_degrees1: array of rotation in degrees
    :param rot_degrees2: array of rotation in degrees
    :return: None
    '''
    c1 = abs(rot_degrees1[1])<2 and abs(rot_degrees1[2])<2
    if rot_degrees2 is not None:
        c2 = abs(rot_degrees2[1])<2 and abs(rot_degrees2[2])<2
        assert(c1 and c2)
    else:
        assert(c1)


def main():
    found_ref_pos = False
    reached_edge = False

    foldername="cam2/"
    ref_image_name = "ref_image.jpg"
    image_name = None

    ref_image = read_image(foldername, ref_image_name)
    # image = read_image(foldername, image)

    # folder_tilt = "cam1/pan360_tilt_0/"
    tilted=0
    panned = 0

    while not found_ref_pos:
        while panned <= 18:
            # images_360 = read_snapshot_360(folder_tilt)
            # for snapshot in images_360:
            print(f"Panning {panned} ...")
            snapshot = take_picture()
            # print("file name is ", new_filename)
            snap_image = read_image(foldername, snapshot)
            num_points_matched, rot_matrices = calculate_rot_matrix(ref_img=ref_image, img=snap_image)
            if num_points_matched>200:
                print(f"Found a good match with {num_points_matched} good points")
                # plt.imshow(image, 'gray'),plt.show()
                # time.sleep(5)
                found_ref_pos = True
                recognized_image = snapshot
                break
            else:
                print(f"Not enough good matches: {num_points_matched}")
            cam.move(float(cam.get_position()["pan"])+20, float(cam.get_position()["tilt"]))
            panned+=1

        if not found_ref_pos:
            # go down until reached bottom, go up until 0 -> scanned everything
            tilted+=1
            panned = 0
            if not reached_edge:
                print(f"Tilting {tilted} ...")
                # folder_tilt = f'pan360_tilt_{tilted}'
                prev_tilt = float(cam.get_position()["tilt"])
                cam.move(float(cam.get_position()["pan"]), float(cam.get_position()["tilt"])-20)
                cur_tilt = float(cam.get_position()["tilt"])
                if prev_tilt<cur_tilt: #reached the bottom
                    print("Reached the bottom")
                    cam.move(cam.get_position()["pan"], cam.get_position()["tilt"]+20)
                    reached_edge = True
            else:
                cam.move(cam.get_position()["pan"], cam.get_position()["tilt"]+20)
                if(tilted==9):
                    raise Exception("Scanned everything, could not find matching snapshot")
    
    # print(recognized_image)
    recognized_image_split = recognized_image.split('_')
    abs_pan = float(recognized_image_split[1][1:])
    abs_tilt = float(recognized_image_split[2][1:][:-4])
    print(cam.move(abs_pan, abs_tilt))
    print(f"Moving to recognized position: pan {abs_pan} tilt {abs_tilt}")
    
    rot1 = rot_matrices[0]
    rot_degrees1 = calculate_degrees(rot1)
    rot_degrees2 = None
    if (len(rot_matrices)>2):
        print("There are 2 choices for rotation")
        rot2 = rot_matrices[2]
        rot_degrees2 = calculate_degrees(rot2)
        max_degree_pan = max(abs(rot_degrees1[1]), abs(rot_degrees1[2]), abs(rot_degrees2[1]), abs(rot_degrees2[2]))
    else:
        print("There is 1 choice for rotation")
        max_degree_pan = max(abs(rot_degrees1[1]), abs(rot_degrees1[2]))

    check_rot_array_pan(rot_degrees1=rot_degrees1, rot_degrees2=rot_degrees2)

    if(rot_degrees1[1]<0):
        print("degree is negative")
        max_degree_pan = -max_degree_pan
    
    print(f"Start with this panning rotation fix {max_degree_pan}")
    prev_max_degree_pan = 360

    while(abs(max_degree_pan)>1):
        print(f"Panning by {max_degree_pan}")
        # panning motion first
        assert(abs(prev_max_degree_pan)>abs(max_degree_pan))
        prev_max_degree_pan = max_degree_pan
        # go the other way
        cam.move(float(cam.get_position()["pan"])-max_degree_pan, float(cam.get_position()["tilt"]))
        new_pan = float(cam.get_position()["pan"])
        new_tilt = float(cam.get_position()["tilt"])
        print(f'New position after pan: pan {new_pan} | tilt {new_tilt}')
        new_filename = take_picture()
        # new_filename = f'{host}_p{new_pan}_t{new_tilt}.jpg'
        print("file name is ", new_filename)
        image_moved = read_image(foldername, new_filename)
        num_points_matched, rot_matrices = calculate_rot_matrix(ref_img=ref_image, img=image_moved)
        rot1 = rot_matrices[0]
        rot2 = rot_matrices[2]
        rot_degrees1 = calculate_degrees(rot1)
        rot_degrees2 = calculate_degrees(rot2)
        
        if (len(rot_matrices)>2):
            print("There are 2 choices for rotation")
            rot2 = rot_matrices[2]
            rot_degrees2 = calculate_degrees(rot2)
            max_degree_pan = max(abs(rot_degrees1[1]), abs(rot_degrees1[2]), abs(rot_degrees2[1]), abs(rot_degrees2[2]))
        else:
            print("There is 1 choice for rotation")
            max_degree_pan = max(abs(rot_degrees1[1]), abs(rot_degrees1[2]))
            
        if(rot_degrees1[1]<0):
            max_degree_pan = -max_degree_pan

        check_rot_array_pan(rot_degrees1=rot_degrees1, rot_degrees2=rot_degrees2)

        
        print(f"Correcting with {max_degree_pan} degree pan")
    
    final_pan = float(cam.get_position()["pan"])
    final_tilt = float(cam.get_position()["tilt"])
    print(f"Done panning. Position is now pan {final_pan} tilt {final_tilt}")

    print(f"Start tilting...")
    if rot_degrees2 is not None:
        max_degree_tilt = max(abs(rot_degrees1[0]), abs(rot_degrees2[0]))
    else:
        max_degree_tilt = max(abs(rot_degrees1[0]))

    check_rot_arr_tilt(rot_degrees1=rot_degrees1, rot_degrees2=rot_degrees2)
    if(rot_degrees1[0]<0):
        max_degree_tilt = -max_degree_tilt

    print(f"Start with this tilting rotation fix {max_degree_tilt}")
    rot1 = None
    rot2 = None
    rot_degrees1 = None
    rot_degrees2 = None
    prev_max_degree_tilt = 360
    while(abs(max_degree_tilt)>.5):
        print(f"Tilting by {max_degree_tilt}")
        assert(abs(prev_max_degree_tilt)>abs(max_degree_tilt))
        prev_max_degree_tilt = max_degree_tilt
        cam.move(float(cam.get_position()["pan"]), float(cam.get_position()["tilt"])-max_degree_tilt)
        new_pan = float(cam.get_position()["pan"])
        new_tilt = float(cam.get_position()["tilt"])
        print(f'New position after tilt: pan {new_pan} | tilt {new_tilt}')

        # new_filename = f'{host}_p{new_pan}_t{new_tilt}.jpg'
        new_filename = take_picture()
        image_moved = read_image(foldername, new_filename)
        num_points_matched, rot_matrices = calculate_rot_matrix(ref_img=ref_image, img=image_moved)
        rot1 = rot_matrices[0]
        rot_degrees1 = calculate_degrees(rot1)
        if (len(rot_matrices)>2):
            rot2 = rot_matrices[2]
            rot_degrees2 = calculate_degrees(rot2)
            max_degree_tilt = max(abs(rot_degrees1[0]), abs(rot_degrees2[0]))
        else:
            max_degree_tilt = max(abs(rot_degrees1[0]))

        check_rot_arr_tilt(rot_degrees1=rot_degrees1, rot_degrees2=rot_degrees2)
        if(rot_degrees1[0]<0):
            max_degree_tilt = -max_degree_tilt

    # finished tilting
    print(f'Finished tilting: position is pan {new_pan} | tilt {new_tilt}')
        
    print("DONE")

if __name__ == '__main__':
    sys.exit(main())