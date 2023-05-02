import sys
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from homography import rotationMatrixToEulerAngles
import csv
import os
import time

# SIFT Constants
SIFT_LOWE_DIST_RATIO = .7
SIFT_MIN_MATCH_COUNT = 10

# BRISK Constants
BRISK_LOWE_DIST_RATIO = .75
BRISK_MIN_MATCH_COUNT = 4

# ORB Constants
ORB_MAX_FEATURES = 500
ORB_GOOD_MATCH_PERCENT = .15
ORB_MIN_MATCH_COUNT = 4

# SURF Constants
SURF_LOWE_DIST_RATIO = .75

# Intinsic Camera Matrix
FOCAL_LEN = 1483.5
PP = (920.5, 571.6)
CAM_INTR = np.array([[1489.6, 0, 920.5],[0, 1476.1, 571.6],[0, 0, 1]])

# Map of camera's home {pan, tilt}
home = {1: {'p':-152.06, 't':-22.6},
        2: {'p':134.02, 't':-43.36},
        3: {'p':120.66, 't':-64.51},
        4: {'p':89.24, 't':-57.64},
        5: {'p':36.01, 't':-38.87},
        6: {'p':-71.45, 't':-26.29}}

def format_to_csv(filename, num_points_matched, M_rot, E_rot1, E_rot2 ):
    data = []
    name_split = filename.split('_')
    cam_num = int(name_split[0][-1:]) #last number of ip address
    abs_pan = float(name_split[1][1:])
    abs_tilt = float(name_split[2][1:][:-4])
    # print('home[cam_num] ', home[cam_num])
    rel_pan = home[cam_num]['p']-abs_pan
    rel_tilt = home[cam_num]['t']-abs_tilt
    data.append(cam_num)
    data.append(abs_pan)
    data.append(abs_tilt)
    data.append(num_points_matched)
    data.append(rel_pan)
    data.append(rel_tilt)

    M_degrees=[]
    for i in range(len(M_rot)):
        M_degrees.append(rot_matrix_to_degrees(M_rot[i]))
    
    E_degrees1 = rot_matrix_to_degrees(E_rot1)
    E_degrees2 = rot_matrix_to_degrees(E_rot2)

    # print(M_degrees)
    M_degrees=np.around(M_degrees, decimals=1)
    E_degrees1=np.around(E_degrees1, decimals=1)
    E_degrees2=np.around(E_degrees2, decimals=1)
    M_degrees1=M_degrees[0]
    if(len(M_degrees)>2):
        M_degrees2=M_degrees[2]
    else:
        M_degrees2=None
    # print(M_degrees1)
    # print(M_degrees2)
    # exit()
    data.append(M_degrees1)
    data.append(M_degrees2)
    data.append(E_degrees1)
    data.append(E_degrees2)
    
    # print(data)

    return data

def load_images_from_folder(folder):
    images=[]
    count=0
    for filename in os.listdir(folder):
        if filename != "home.jpg" and filename[-3:]=="jpg" :
            images.append(filename)
            count+=1
    print(f'loaded {count} images from {folder}')
    return images

def read_images(filename, ref_file_name, img_file_name):
     # Training image
    print(ALGO+": Reading reference image: ", filename+ref_file_name)
    ref_img = cv.imread(filename+ref_file_name, cv.COLOR_BGR2GRAY)

    # Query image
    print(ALGO+": Reading image to align: ", filename+img_file_name)
    img = cv.imread(filename+img_file_name, cv.COLOR_BGR2GRAY)

    return ref_img, img

def find_good_matches(matches, kp_img, kp_rimg, ratio, match_count):
    # Store all the good matches as per Lowe's ratio test
    good = []
    for m,n in matches:
        if m.distance < ratio*n.distance:
            good.append(m)

    # Found enough matches, success
    if len(good)>match_count:
        print(ALGO+": Success!" ) 
        dst_pts = np.float32([ kp_rimg[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        src_pts = np.float32([ kp_img[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    else:
        raise Exception(ALGO+": Not enough matches are found - {}/{}".format(len(good), SIFT_MIN_MATCH_COUNT)) 
   
    return dst_pts, src_pts, good

def rot_matrix_to_degrees(rot):
    euler_angles = rotationMatrixToEulerAngles(rot)
    # print(str(i)+" eurler angles: ", euler_angles)
    degrees_angles = euler_angles
    for j in range(3):
        degrees_angles[j]=degrees_angles[j]*180/3.1415
    return degrees_angles

def draw_and_save_lines(M, mask, img, img_reference, kp_img, kp_rimg, good, out_file_postfix):
    matchesMask = mask.ravel().tolist()
    h,w = img.shape[:-1]
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv.perspectiveTransform(pts,M)
    img_reference = cv.polylines(img_reference,[np.int32(dst)],True,255,3, cv.LINE_AA)

    draw_params = dict(#matchColor = Scalar::all(-1), # draw matches in random color
            singlePointColor = None,
            matchesMask = matchesMask, # draw only inliers
            flags = 2)
    print("",len(good))
    img3 = cv.drawMatches(img_reference,kp_rimg,img,kp_img,good,None,**draw_params)
    # plt.imshow(img3, 'gray'),plt.show()
    # sleep(1)

    # Write match image to disk
    out_file_name = f'{ALGO}_matchlines_{out_file_postfix}.png'
    print(ALGO+": Saving match image: ", out_file_name)
    cv.imwrite(out_file_name, img3)

def save_planar_transform(M, img, img_reference, out_file_postfix):
    # Use homography
    height, width, channels = img_reference.shape
    img_aligned = cv.warpPerspective(img, M, (width, height))

    # Write aligned image to disk
    out_file_name = ALGO+"_aligned"+str(out_file_postfix)+".png"
    print(ALGO+" Saving aligned image: ", out_file_name)
    cv.imwrite(out_file_name, img_aligned)

def print_homography_result(M, rot, tran):
    print("homography: \n", M)
    print("\n")
    for i in range(4):
        degrees_angles = rot_matrix_to_degrees(rot[i])
        print("rotation "+str(i)+" in degrees: ", degrees_angles)
        print("\n")
        # for i in range(4):
            # print("translation "+str(i)+": ", tran[i])

def print_essential_result(E, rot, tran):
    print("Essential Matrix: \n", E)
    print("\n")
    degrees_angles = rot_matrix_to_degrees(rot)
    print("rotation in degrees: ", degrees_angles)
    # print("translation: ", tran[0])

def sift_algo(filename, ref_file_name, img_file_name, out_file_postfix):
    # Read in training and query image
    ref_img, img = read_images(filename, ref_file_name, img_file_name)
    
    # Initiate SIFT detector
    sift = cv.SIFT_create()

    # Find the keypoints and descriptors with SIFT
    kp_rimg, descr_rimg = sift.detectAndCompute(ref_img,None)
    kp_img, descr_img = sift.detectAndCompute(img,None)

    # Matching with FLANN
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    matcher = cv.FlannBasedMatcher(index_params, search_params)
    matches = matcher.knnMatch(descr_rimg,descr_img,k=2) #finding best 2 matches to use for Lowe's test

    # Returns list of "good" matches 
    dst_pts, src_pts, good = find_good_matches(matches, kp_img, kp_rimg, SIFT_LOWE_DIST_RATIO, SIFT_MIN_MATCH_COUNT)
   
    # Calculate essential matrix
    E, E_mask = cv.findEssentialMat(points1=dst_pts, points2=src_pts, cameraMatrix=CAM_INTR)
    # E_points, E_rot, E_tran, E_mask = cv.recoverPose(E, points1=dst_pts, points2=src_pts)
    E_rot1, E_rot2, E_tran = cv.decomposeEssentialMat(E=E) 
    # print_essential_result(E, E_rot, E_tran)

    # Calculate homography
    M, M_mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
    M_ret, M_rot, M_tran, M_norm = cv.decomposeHomographyMat(M, CAM_INTR)
    # print_homography_result(M, M_rot, M_tran)

    # Saves image of matches drawn and image that has been aligned with transformation
    
    draw_and_save_lines(M, M_mask, img, ref_img, kp_img, kp_rimg, good, out_file_postfix+'M')
    # draw_and_save_lines(E, E_mask, img, ref_img, kp_img, kp_rimg, good, out_file_postfix+'E')
    # save_planar_transform(M, img, ref_img, out_file_postfix)

    return M_rot, E_rot1, E_rot2, len(good)


def brisk_algo(filename, ref_file_name, img_file_name, out_file_postfix):
    # Read in training and query image
    ref_img, img = read_images(filename, ref_file_name, img_file_name)
    
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
    
    # Returns list of "good" matches 
    dst_pts, src_pts, good = find_good_matches(matches, kp_img, kp_rimg, BRISK_LOWE_DIST_RATIO, BRISK_MIN_MATCH_COUNT)
    
    # Calculate essential matrix
    E, mask = cv.findEssentialMat(points1=dst_pts, points2=src_pts, cameraMatrix=CAM_INTR)
    # E_points, E_rot, E_tran, E_mask = cv.recoverPose(E, points1=dst_pts, points2=src_pts)
    E_rot1, E_rot2, E_tran = cv.decomposeEssentialMat(E=E) 
    # print_essential_result(E, E_rot, E_tran)

    # Calculate homography
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
    M_ret, M_rot, M_tran, M_norm = cv.decomposeHomographyMat(M, CAM_INTR)
    # print_homography_result(M, M_rot, M_tran)


    # Saves image of matches drawn and image that has been aligned with transformation
    # draw_and_save_lines(M, mask, img, ref_img, kp_img, kp_rimg, good, out_file_postfix)
    # draw_and_save_lines(E, mask, img, ref_img, kp_img, kp_rimg, good, out_file_postfix)

    # save_planar_transform(M, img, ref_img, out_file_postfix)

    return M_rot, E_rot1, E_rot2, len(good)


def orb_algo(filename, ref_file_name = "ref_image.jpg", img_file_name = "image.jpg", out_file_postfix=""):
    # Read in training and query image
    ref_img, img = read_images(filename, ref_file_name, img_file_name)
    
    # Detect ORB features and compute decriptors
    orb = cv.ORB_create(ORB_MAX_FEATURES)

    # Find the keypoints and descriptors with ORB
    kp_rimg, descr_rimg = orb.detectAndCompute(ref_img, None)
    kp_img, descr_img = orb.detectAndCompute(img, None)

    # Match features with brute force
    matcher = cv.DescriptorMatcher_create(cv.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING) #cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descr_rimg, descr_img, None)

    # Returns list of "good" matches 
    matches = sorted(matches, key=lambda x: x.distance, reverse=False)
    good_matches_count = int(len(matches) * ORB_GOOD_MATCH_PERCENT)
    good = matches[:good_matches_count]

    # Found enough matches, success
    if len(good)>ORB_MIN_MATCH_COUNT:
        print(ALGO+": Success!" ) 
        dst_pts = np.float32([ kp_rimg[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        src_pts = np.float32([ kp_img[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    else:
        raise Exception(ALGO+": Not enough matches are found - {}/{}".format(len(good), ORB_MIN_MATCH_COUNT)) 
      
    # Calculate essential matrix
    E, mask = cv.findEssentialMat(points1=dst_pts, points2=src_pts, focal=FOCAL_LEN, pp=PP)
    # e_points, E_rot, E_tran, mask = cv.recoverPose(E, points1=dst_pts, points2=src_pts)
    E_rot1, E_rot2, E_tran = cv.decomposeEssentialMat(E=E) 
    # print_essential_result(E, E_rot, E_tran)

    # Find homography
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC) # srcPoints, dstPoints
    M_ret, M_rot, M_tran, M_norm = cv.decomposeHomographyMat(M, CAM_INTR)
    # print_homography_result(M, M_rot, M_tran)

    draw_and_save_lines(M, mask, img, ref_img, kp_img, kp_rimg, good, out_file_postfix)
    # save_planar_transform(M, img, ref_img, out_file_postfix)

    return M_rot, E_rot1, E_rot2, len(good)

def surf_algo(filename, ref_file_name = "ref_image.jpg", img_file_name = "image.jpg", out_file_postfix=""):
    # Read in training and query image
    ref_img, img = read_images(filename, ref_file_name, img_file_name)

    # Detect ORB features and compute decriptors
    surf = cv.SURF_create()

    # Find the keypoints and descriptors with ORB
    kp_rimg, descr_rimg = surf.detectAndCompute(ref_img, None)
    kp_img, descr_img = surf.detectAndCompute(img, None)

    # Match features with brute force
    matcher = cv.BFMatcher(cv.NORM_L2)
    matches = matcher.matcher.knnMatch(descr_rimg, trainDescriptors = descr_img, k = 2) #2

    # Returns list of "good" matches 
    dst_pts, src_pts, good = find_good_matches(matches, kp_img, kp_rimg, SURF_LOWE_DIST_RATIO, 0)

    # Calculate essential matrix
    E, mask = cv.findEssentialMat(points1=dst_pts, points2=src_pts, focal=FOCAL_LEN, pp=PP)
    e_points, E_rot, E_tran, mask = cv.recoverPose(E, points1=dst_pts, points2=src_pts)
    # print_essential_result(E, E_rot, E_tran)

    # Find homography
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC) # srcPoints, dstPoints
    M_ret, M_rot, M_tran, M_norm = cv.decomposeHomographyMat(M, CAM_INTR)
    # print_homography_result(M, M_rot, M_tran)

    draw_and_save_lines(M, mask, img, ref_img, kp_img, kp_rimg, good, out_file_postfix)
    # save_planar_transform(M, img, ref_img, out_file_postfix)

    return M_rot, E_rot

def main():
    # Setting up
    # test='10.80.134.61_p-162.20_t-22.60.jpg'
    # print(test[11:][:-4])
    # exit()
    global ALGO
    ref_filename = "home.jpg"
    foldername= "cam6"
    image_names=load_images_from_folder(foldername)
    out_file_postfix = ""

    ALGO = input("Which feature detection algorithm to use? ")

    header = ['cam_num', 'abs_pan', 'abs_tilt', 'num_points_matched', 'rel_pan', 'rel_tilt', 'M_rot1','M_rot2', 'E_rot1', 'E_rot2']
    with open(f'{ALGO}_{foldername}_results.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        start = time.time()
    
        if(ALGO=="all"):
            print("Starting SIFT algorithm...")
            for image_name in image_names:
                M_rot, E_rot1, E_rot2, num_good = sift_algo(filename=f'{foldername}/', ref_file_name=ref_filename, img_file_name=image_name, out_file_postfix=image_name[11:][:-4])
                csv_data = format_to_csv(image_name, num_good, M_rot, E_rot1, E_rot2 )
                writer.writerow(csv_data)

            print("Starting ORB algorithm...")
            for image_name in image_names:
                M_rot, E_rot1, E_rot2, num_good = orb_algo(filename=f'{foldername}/', ref_file_name=ref_filename, img_file_name=image_name, out_file_postfix=image_name[11:][:-4])
                csv_data = format_to_csv(image_name, num_good, M_rot, E_rot1, E_rot2 )
                writer.writerow(csv_data)

            print("Starting BRISK algorithm...")
            for image_name in image_names:
                M_rot, E_rot1, E_rot2, num_good = brisk_algo(filename=f'{foldername}/', ref_file_name=ref_filename, img_file_name=image_name, out_file_postfix=image_name[11:][:-4])
                csv_data = format_to_csv(image_name,num_good, M_rot, E_rot1, E_rot2 )
                writer.writerow(csv_data)

        if(ALGO == "sift"):
            print("Starting SIFT algorithm...")
            for image_name in image_names:
                M_rot, E_rot1, E_rot2, num_good = sift_algo(filename=f'{foldername}/', ref_file_name=ref_filename, img_file_name=image_name, out_file_postfix=image_name[11:][:-4])
                csv_data = format_to_csv(image_name, num_good, M_rot, E_rot1, E_rot2 )
                writer.writerow(csv_data)

        # elif (ALGO=="orb"):
        #     print("Starting ORB algorithm...")
        #     for image_name in image_names:
        #         M_rot, E_rot1, E_rot2, num_good = orb_algo(filename=f'{foldername}/', ref_file_name=ref_filename, img_file_name=image_name, out_file_postfix=image_name[11:][:-4])
                # csv_data = format_to_csv(image_name, M_rot, E_rot)
                # writer.writerow(csv_data)

        elif (ALGO=="brisk"):
            print("Starting BRISK algorithm...")
            for image_name in image_names:
                M_rot, E_rot1, E_rot2, num_good = brisk_algo(filename=f'{foldername}/', ref_file_name=ref_filename, img_file_name=image_name, out_file_postfix=image_name[11:][:-4])
                csv_data = format_to_csv(image_name,num_good, M_rot, E_rot1, E_rot2 )
                writer.writerow(csv_data)

        # elif (ALGO=="surf"):
        #     print("Starting SURF algorithm...")
        #     for image_name in image_names:
        #         M_rot, E_rot = surf_algo(filename=f'{foldername}/', ref_file_name=ref_filename, img_file_name=image_name, out_file_postfix=image_name[11:][:-4])
        #         csv_data = format_to_csv(image_name, M_rot, E_rot1, E_rot2 )
        #         writer.writerow(csv_data)

        else:
            print("Invalid")

        end = time.time()
        print(f'{ALGO} time taken: {end-start}')

if __name__ == '__main__':
    # print(cv.__version__)
    # exit()
    sys.exit(main())