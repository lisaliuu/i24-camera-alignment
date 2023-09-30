# Dynamic Camera Alignment

## Python script that dynamically aligns a camera to a reference image with OpenCV.
This Python script is developed for the [I24-MOTION testbed](https://i24motion.org/) data processing pipeline. 

## Motivation
1. To research the use of OpenCV in feature detection of highway images.
2. To investigate the homography and essential matrix calculated and its validity.
3. To enable dynamic alignment of cameras to a certain view of the highway instead of static alignment to a hard-coded angle. Dynamic camera alignment reduces video processing error as poles supporting the cameras may distort over time.

## Files

### 1. `feature_detection.py`
Compares different feature detection algorithms and calculates rotation arrays of sample highway images and a reference image.

#### Input:
- reference image named `home.jpg` (line 338)
- folder path where reference image and query images (to match) are stored (line 339)

#### Output:
CSV file with have the following columns: `['cam_num', 'abs_pan', 'abs_tilt', 'num_points_matched', 'rel_pan', 'rel_tilt', 'M_rot1','M_rot2', 'E_rot1', 'E_rot2']`

### 2. `homography.py`
Tests the calculation of homography matrix and rotation matrix by calibrating the camera with chessboard images.

#### Input:
- reference image named `ref_image.jpg` (line 66)
- query image named `image.jpg` (line 66)
- chessboard images in format `CAM_*.jpg` (line 109)

#### Output:
Prints the camera, homography, rotation, translation, and normals matrix.

### 3. `camera_alignment.py`
Dynamic camera alignment script.
#### Algorithm: 
![camaligndiagram](https://github.com/lisaliuu/i24-camera-alignment/assets/82255401/177a1f4b-c5de-48db-9853-b114ed9cad90)

**(lines 257 - 303):** The camera pans to search for a view that has > 200 feature points matches by taking pictures of the highway at every 20°, tilting by 20° if the camera has been panned 360° at a certain tilt. Once > 200 feature points are found, the homography and rotation matrix is calculated of that camera view to the reference image. 

**(lines 303 - 354):** The camera pans according to the angle from the rotation matrix (y, z axis). The homography is repeatedly calculated **(line 336)** and the camera panned accordingly until the panning angle difference is < 1°. 

**(lines 354 - 392):** The camera adjusts its tilt according to the rotation matrix (x axis) by repeatedly calculating homography **(line 378)** until the tilt angle difference is < 0.5°.

#### Assumptions:
##### The following assumptions are made based on [data](cam_rot_results.xlsx) gathered on the calculated rotation matrix from the 6 cameras.

- [OpenCV's](https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga7f60bdff78833d1e3fd6d9d0fd538d92) `decomposeHomographyMat()` returns 4 rotation matrices (unless ref_image is the same as camera view, then one array ([0, 0, 0]) is returned), of which the 1st and 2nd matrices are the same, and 3rd and 4th matrices are the same.
- 200 matched or "good" feature points produces valid homography
- The first element of rotation array (x axis) is used for tilting, the second and third element (y and z axis) are used for panning
- In the rotation arrays, the angle with the largest magnitude is the closest to the actual panning/tilting angle difference
- Panning correction is performed first as the rotation array from pure panning produces non-negligible x axis angle value but rotation array from pure tilting has negligble y and z axis angle value.

#### Input:
- credentials (lines 18 - 21)
- path to save snapshots (line 37)
- reference image path (line 250)

#### Output:
An aligned camera.

#### Notes:
- The naming of snapshots created as camera is scanning and parsed is in the format: `f'{host}_p{pan}_t{tilt}.jpg'`, i.e. `10.80.134.66_p-71.45_t-26.29.jpg`
