# camera_alignment.py

Aligns camera with a given reference image using homography and BRISK feature detection in OpenCV.

## Algorithm: 

**(lines 257 - 303):** First, the camera pans to search for a view that has > 200 feature points matches by taking pictures of the highway at every 20°, tilting by 20° if the camera has been panned 360° at a certain tilt. Once > 200 feature points are found, the homography and rotation matrix is calculated of that camera view to the reference image. 

**(lines 303 - 354):** Then, the camera is panned according to the angle from the rotation matrix (y, z axis). The homography is repeatedly calculated **(line 336)** and the camera panned accordingly until the panning angle difference between the current camera view and reference image is < 1°. 

**(lines 354 - 392):** Finally, the camera adjusts its tilt according to the rotation matrix (x axis) by repeatedly calculating homography **(line 378)** until the tilt angle difference is < 0.5°.

## Assumptions:
##### The following assumptions are made based on data gathered on the calculated rotation matrix from the 6 cameras.

- [OpenCV's](https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga7f60bdff78833d1e3fd6d9d0fd538d92) `decomposeHomographyMat()` returns 4 rotation matrices (unless ref_image is the same as camera view, then one array ([0, 0, 0]) is returned), of which the 1st and 2nd matrices are the same, and 3rd and 4th matrices are the same.
- 200 matched or "good" feature points produces valid homography
- The first element of rotation array (x axis) is used for tilting, the second and third element (y and z axis) are used for panning
- In the rotation arrays, the angle with the largest magnitude is the closest to the actual panning/tilting angle difference
- Panning correction is performed first as the rotation array from pure panning produces non-negligible x axis angle value but rotation array from pure tilting has negligble y and z axis angle value.


## Usage:
Modify:

- **(lines 18 - 21):** credentials
- **(line 37):** path to save snapshots that the camera takes
- **(line 250):** reference image path

## Misc.:
- Snapshots written as camera is scanning and parsed when reading is in the format: `f'{host}_p{pan}_t{tilt}.jpg'`, i.e. `10.80.134.66_p-71.45_t-26.29.jpg`


# feature_detection.py
Developed to compare different feature detection algorithms and collect data. Records data in CSV file of the performance of different feature detection algorithms and their calculated rotation arrays from homography and essential matrix of a series of query images to a reference image.

## CSV Format
Generated CSV file will have the following information: `['cam_num', 'abs_pan', 'abs_tilt', 'num_points_matched', 'rel_pan', 'rel_tilt', 'M_rot1','M_rot2', 'E_rot1', 'E_rot2']`

## Usage:
Must have: 

- **(line 338):** reference image named `home.jpg`
- **(line 339):** define folder path where reference image and query images (to match) are stored
	- All images are in `.jpg` format

# homography.py
Developed to test calculating homography, calibrating camera, and finding rotation matrix


## Usage:
Must have: 

- **(line 66):** reference image named `ref_image.jpg`
- **(line 66):** query image named `image.jpg`
- **(line 109):** chessboard images in format `CAM_*.jpg`
	- All images are in `.jpg` format