import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import pathlib
import math

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = .15 #idk, copied from online

def align_images(img, img_reference, out_file_postfix):
    '''
    Calculates homography using ORB feature detection and writes matches and warped query image
    '''
    # Make imgs gray for `ORB_create`
    img1_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img_reference, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute decriptors
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(img1_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2_gray, None)

    # Match features
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING) #cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2, None)

    # print("Matches: ",matches)
    # Sort matches
    matches = sorted(matches, key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    good_matches_count = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:good_matches_count]

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt # img1
        points2[i, :] = keypoints2[match.trainIdx].pt # img2

    # Find homography
    M, mask = cv2.findHomography(points1, points2, cv2.RANSAC) # srcPoints, dstPoints

    # Draw good matches
    h,w = img.shape[:-1]
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    img_reference = cv2.polylines(img_reference,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    matchesMask = mask.ravel().tolist()
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
        singlePointColor = None,
        matchesMask = matchesMask, # draw only inliers
        flags = 2)
    img_matches = cv2.drawMatches(img_reference, keypoints2,img, keypoints1,matches, None, **draw_params)
    cv2.imwrite("orb_matches"+out_file_postfix+".jpg", img_matches)
    # plt.imshow(img_matches),plt.show()

    # Use homography
    img_aligned = cv2.warpPerspective(img, M, (w, h))
    return img_aligned, M

def homography(ref_file_name = "ref_image.jpg", img_file_name = "image.jpg", out_file_postfix=""):
    '''
    Reads in files for align_images
    '''
    # Read the file image
    print("Reading reference image: ", ref_file_name)
    img_reference = cv2.imread(ref_file_name, cv2.IMREAD_COLOR) # Always convert image to the 3 channel BGR color image

    # Read image to be aligned
    print("Reading image to align: ", img_file_name)
    img = cv2.imread(img_file_name, cv2.IMREAD_COLOR) # img is a NumPy N-dimensional array

    if img is None or img_reference is None:
        print("Error reading in images.")
    else:
        print("Aligning images ...")

        # img_aligned is img after applying h, homography
        img_aligned, h = align_images(img, img_reference, out_file_postfix)

        # Write aligned image to disk
        out_file_name = "orb_aligned"+out_file_postfix+".jpg"
        print("Saving aligned image: ", out_file_name)
        cv2.imwrite(out_file_name, img_aligned)

        return h

def calibrate_chessboard(width, height):
    '''
    Calculates camera intrinsic matrix based on chessboard pictures
    '''
    success = True
    # termination criteria, number of iterations completed or accuracy is reached
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((height*width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    
    images = glob.glob('CAM_*.JPG')
    print(images)
    if not images:
        print("Error reading in images.")
    else:
        for filename in images:
            print(filename)
            img = cv2.imread(str(filename))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (width,height), None)

            # If corners were found, add object and refined image points
            if ret:
                print("Corners were found")
                # Standard coordinates of a chessboard (0, 0, 0), (1, 0, 0), etc
                objpoints.append(objp)

                # Refines corner locations
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)

                # Draw and display the corners
                # cv2.drawChessboardCorners(img, (width,height), corners2, ret)
                # cv2.imshow('img', img)
                # cv2.waitKey(800)

                
            else:
                print("Corners were not found")
                success = False

            cv2.destroyAllWindows()
    if success:
            print("Calibrating camera...")
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
            # print("ret: %f", ret)
            # print("mtx: %f", mtx)
            # print("dist: %f", dist)
            # print("rvecs: %f", rvecs)
            # print("tvecs: %f", tvecs)

            return mtx

    
def isRotationMatrix(R) :
    '''
    Checks if a matrix is a valid rotation matrix
    '''
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
 
def rotationMatrixToEulerAngles(R) :
    '''
    Calculates rotation matrix to euler angles
    The result is the same as MATLAB except the order
    of the euler angles ( x and z are swapped ).
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


def main():
    int_cam_mtx = calibrate_chessboard (9, 6)
    homogr = homography()
    if int_cam_mtx is not None and homogr is not None:
        print("camera matrix: ", int_cam_mtx)
        print("homography: ", homogr)
        ret, rot, tran, norm = cv2.decomposeHomographyMat(homogr, int_cam_mtx)
        print("ret: ", ret)
        print("rotation: ", rot)
        print("translation: ", tran)
        print("normals: ", norm)

        euler_angles = rotationMatrixToEulerAngles(rot[0])
        print("eurler angles: ", euler_angles)
        degrees_angles = euler_angles
        for i in range(3):
            degrees_angles[i]=degrees_angles[i]*180/3.1415
        print("in degrees: ", degrees_angles)
    else:
        print("Error in decomposing homography")

if __name__ == '__main__':
    sys.exit(main())