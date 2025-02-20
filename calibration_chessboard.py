import cv2
import numpy as np
import glob

# ----------------------------
# Calibration Pattern Settings
# ----------------------------
chessboard_size = (7, 10)  
square_size = 0.025       


objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size


objpoints = []           
imgpoints_left = []      
imgpoints_right = []     

# ----------------------------
# Load Calibration Images
# ----------------------------
images_left = sorted(glob.glob('left/*.jpg'))
images_right = sorted(glob.glob('right/*.jpg'))

if len(images_left) != len(images_right):
    raise ValueError("The number of left and right calibration images must be equal.")


if len(images_left) == 0:
    raise ValueError("No calibration images found in the 'left' folder.")

example_img = cv2.imread(images_left[0])
gray_example = cv2.cvtColor(example_img, cv2.COLOR_BGR2GRAY)
image_shape = gray_example.shape[::-1]  

# ----------------------------
# Detect Chessboard Corners in All Pairs
# ----------------------------
for fname_left, fname_right in zip(images_left, images_right):
    img_left = cv2.imread(fname_left)
    img_right = cv2.imread(fname_right)

    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    ret_left, corners_left = cv2.findChessboardCorners(gray_left, chessboard_size, None)
    ret_right, corners_right = cv2.findChessboardCorners(gray_right, chessboard_size, None)

    if ret_left and ret_right:
        objpoints.append(objp.copy())


        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
        corners_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)

        imgpoints_left.append(corners_left)
        imgpoints_right.append(corners_right)

        cv2.drawChessboardCorners(img_left, chessboard_size, corners_left, ret_left)
        cv2.drawChessboardCorners(img_right, chessboard_size, corners_right, ret_right)
        cv2.imshow('Left Calibration', img_left)
        cv2.imshow('Right Calibration', img_right)
        cv2.waitKey(500)
    else:
        print(f"Chessboard corners not found in pair: {fname_left} and {fname_right}")

cv2.destroyAllWindows()

if not objpoints:
    raise ValueError("No valid chessboard detections were found. Check your calibration images.")

# ----------------------------
# Single Camera Calibration
# ----------------------------
ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
    objpoints, imgpoints_left, image_shape, None, None)
ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
    objpoints, imgpoints_right, image_shape, None, None)

print("Left Camera Matrix:\n", mtx_left)
print("Right Camera Matrix:\n", mtx_right)

# ----------------------------
# Stereo Calibration
# ----------------------------
flags = cv2.CALIB_FIX_INTRINSIC
criteria_stereo = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)

ret_stereo, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_left, imgpoints_right,
    mtx_left, dist_left, mtx_right, dist_right,
    image_shape,
    criteria=criteria_stereo, flags=flags
)

print("Rotation Matrix between cameras:\n", R)
print("Translation Vector between cameras:\n", T)

# ----------------------------
# Stereo Rectification
# ----------------------------
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    mtx_left, dist_left, mtx_right, dist_right,
    image_shape, R, T, alpha=0
)

map1_left, map2_left = cv2.initUndistortRectifyMap(
    mtx_left, dist_left, R1, P1, image_shape, cv2.CV_16SC2)
map1_right, map2_right = cv2.initUndistortRectifyMap(
    mtx_right, dist_right, R2, P2, image_shape, cv2.CV_16SC2)

# ----------------------------
# Save Calibration Parameters
# ----------------------------
np.savez("stereo_calibration.npz",
         mtx_left=mtx_left, dist_left=dist_left,
         mtx_right=mtx_right, dist_right=dist_right,
         R=R, T=T, E=E, F=F,
         R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
         map1_left=map1_left, map2_left=map2_left,
         map1_right=map1_right, map2_right=map2_right)

print("Stereo calibration complete. Parameters saved to 'stereo_calibration.npz'.")
