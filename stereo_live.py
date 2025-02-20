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

def main():
    # Load calibration parameters
    (mtx_left, dist_left, mtx_right, dist_right, 
     R1, R2, P1, P2, Q, map1_left, map2_left, map1_right, map2_right) = load_calibration_parameters()
    
    
    cap_left = cv2.VideoCapture(1)
    cap_right = cv2.VideoCapture(2)
    
    if not cap_left.isOpened() or not cap_right.isOpened():
        print("Error: Unable to open one or both cameras.")
        return

    
    min_disp = 0
    num_disp = 16 * 5  
    block_size = 7
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * 3 * block_size ** 2,
        P2=32 * 3 * block_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )
    
    print("Press 'q' to exit the live stream.")

    while True:

        ret_left, frame_left = cap_left.read()
        ret_right, frame_right = cap_right.read()
        
        if not ret_left or not ret_right:
            print("Error: Could not capture frames.")
            break


        rect_left = cv2.remap(frame_left, map1_left, map2_left, cv2.INTER_LINEAR)
        rect_right = cv2.remap(frame_right, map1_right, map2_right, cv2.INTER_LINEAR)

        gray_left = cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY)

        disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
        
        disp_vis = cv2.normalize(disparity, None, alpha=0, beta=255,
                                 norm_type=cv2.NORM_MINMAX)
        disp_vis = np.uint8(disp_vis)

        cv2.imshow("Rectified Left", rect_left)
        cv2.imshow("Rectified Right", rect_right)
        cv2.imshow("Live Disparity Map", disp_vis)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
