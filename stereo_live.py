import cv2
import numpy as np

def load_calibration_parameters():
    """
    Load stereo calibration parameters from the saved .npz file.
    """
    calib_data = np.load('stereo_calibration.npz')
    mtx_left   = calib_data['mtx_left']
    dist_left  = calib_data['dist_left']
    mtx_right  = calib_data['mtx_right']
    dist_right = calib_data['dist_right']
    R1         = calib_data['R1']
    R2         = calib_data['R2']
    P1         = calib_data['P1']
    P2         = calib_data['P2']
    Q          = calib_data['Q']
    map1_left  = calib_data['map1_left']
    map2_left  = calib_data['map2_left']
    map1_right = calib_data['map1_right']
    map2_right = calib_data['map2_right']
    return mtx_left, dist_left, mtx_right, dist_right, R1, R2, P1, P2, Q, map1_left, map2_left, map1_right, map2_right

def update_disparity(val=None):
    """
    Update and display the disparity map based on GUI trackbar values.
    """
    global stereo, cap_left, cap_right
    
    # Get current trackbar values
    num_disp = cv2.getTrackbarPos('numDisparities', 'Disparity Settings') * 16
    block_size = cv2.getTrackbarPos('blockSize', 'Disparity Settings')
    if block_size % 2 == 0:  # Ensure odd block size
        block_size += 1
    P1 = cv2.getTrackbarPos('P1', 'Disparity Settings') * 2
    P2 = cv2.getTrackbarPos('P2', 'Disparity Settings') * 2
    disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff', 'Disparity Settings')
    uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio', 'Disparity Settings')
    speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize', 'Disparity Settings')
    speckleRange = cv2.getTrackbarPos('speckleRange', 'Disparity Settings')

    # Update StereoSGBM parameters
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=max(num_disp, 16),  # Must be > 0 and multiple of 16
        blockSize=block_size,
        P1=P1,
        P2=P2,
        disp12MaxDiff=disp12MaxDiff,
        uniquenessRatio=uniquenessRatio,
        speckleWindowSize=speckleWindowSize,
        speckleRange=speckleRange
    )

def main():
    global stereo, cap_left, cap_right

    # Load calibration parameters
    (mtx_left, dist_left, mtx_right, dist_right, 
     R1, R2, P1, P2, Q, map1_left, map2_left, map1_right, map2_right) = load_calibration_parameters()
    
    # Open video capture for both cameras (adjust indices if necessary)
    cap_left = cv2.VideoCapture(0)
    cap_right = cv2.VideoCapture(2)
    
    if not cap_left.isOpened() or not cap_right.isOpened():
        print("Error: Unable to open one or both cameras.")
        return

    # Create initial StereoSGBM object
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=16 * 5,  
        blockSize=7,
        P1=8 * 3 * 7 ** 2,
        P2=32 * 3 * 7 ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )

    # Create GUI window with trackbars
    cv2.namedWindow('Disparity Settings')

    cv2.createTrackbar('numDisparities', 'Disparity Settings', 5, 16, update_disparity)
    cv2.createTrackbar('blockSize', 'Disparity Settings', 7, 50, update_disparity)
    cv2.createTrackbar('P1', 'Disparity Settings', 8 * 3, 100, update_disparity)
    cv2.createTrackbar('P2', 'Disparity Settings', 32 * 3, 200, update_disparity)
    cv2.createTrackbar('disp12MaxDiff', 'Disparity Settings', 1, 25, update_disparity)
    cv2.createTrackbar('uniquenessRatio', 'Disparity Settings', 10, 50, update_disparity)
    cv2.createTrackbar('speckleWindowSize', 'Disparity Settings', 100, 200, update_disparity)
    cv2.createTrackbar('speckleRange', 'Disparity Settings', 32, 50, update_disparity)

    print("Press 'q' to exit the live stream.")

    while True:
        # Capture frames from both cameras
        ret_left, frame_left = cap_left.read()
        ret_right, frame_right = cap_right.read()
        
        if not ret_left or not ret_right:
            print("Error: Could not capture frames.")
            break

        # Rectify the images using precomputed maps
        rect_left = cv2.remap(frame_left, map1_left, map2_left, cv2.INTER_LINEAR)
        rect_right = cv2.remap(frame_right, map1_right, map2_right, cv2.INTER_LINEAR)

        # Convert to grayscale for disparity computation
        gray_left = cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY)

        # Compute the disparity map
        disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
        
        # Normalize disparity for visualization
        disp_vis = cv2.normalize(disparity, None, alpha=0, beta=255,
                                 norm_type=cv2.NORM_MINMAX)
        disp_vis = np.uint8(disp_vis)

        # Display the rectified images and the disparity map
        cv2.imshow("Rectified Left", rect_left)
        cv2.imshow("Rectified Right", rect_right)
        cv2.imshow("Live Disparity Map", disp_vis)

        # Exit loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources and close windows
    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
