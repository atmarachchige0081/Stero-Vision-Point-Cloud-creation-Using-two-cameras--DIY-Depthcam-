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
    R          = calib_data['R']
    T          = calib_data['T']
    R1         = calib_data['R1']
    R2         = calib_data['R2']
    P1         = calib_data['P1']
    P2         = calib_data['P2']
    Q          = calib_data['Q']
    map1_left  = calib_data['map1_left']
    map2_left  = calib_data['map2_left']
    map1_right = calib_data['map1_right']
    map2_right = calib_data['map2_right']
    return (mtx_left, dist_left, mtx_right, dist_right, R, T, 
            R1, R2, P1, P2, Q, map1_left, map2_left, map1_right, map2_right)

def compute_disparity(img_left, img_right):
    """
    Compute disparity map from rectified grayscale stereo images.
    """

    min_disp    = 0
    num_disp    = 16 * 5   
    block_size  = 7

    stereo = cv2.StereoSGBM_create(
        minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize = block_size,
        P1 = 8 * 3 * block_size ** 2,
        P2 = 32 * 3 * block_size ** 2,
        disp12MaxDiff = 1,
        uniquenessRatio = 10,
        speckleWindowSize = 100,
        speckleRange = 32
    )
    disparity = stereo.compute(img_left, img_right).astype(np.float32) / 16.0
    return disparity

def main():

    (mtx_left, dist_left, mtx_right, dist_right, R, T, 
     R1, R2, P1, P2, Q, map1_left, map2_left, map1_right, map2_right) = load_calibration_parameters()


    img_left = cv2.imread("left/left_000.jpg")
    img_right = cv2.imread("right/right_000.jpg")

    if img_left is None or img_right is None:
        print("Error: Could not load test images.")
        return

    gray_left  = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    rect_left  = cv2.remap(gray_left, map1_left, map2_left, cv2.INTER_LINEAR)
    rect_right = cv2.remap(gray_right, map1_right, map2_right, cv2.INTER_LINEAR)

    cv2.imshow("Rectified Left", rect_left)
    cv2.imshow("Rectified Right", rect_right)
    cv2.waitKey(500)  

    disparity = compute_disparity(rect_left, rect_right)

    disp_vis = cv2.normalize(disparity, None, alpha=0, beta=255,
                             norm_type=cv2.NORM_MINMAX)
    disp_vis = np.uint8(disp_vis)
    cv2.imshow("Disparity Map", disp_vis)
    cv2.waitKey(0)

    points_3D = cv2.reprojectImageTo3D(disparity, Q)
    
    mask = disparity > disparity.min()
    out_points = points_3D[mask]
    out_colors = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)[mask]

    save_point_cloud("point_cloud.ply", out_points, out_colors)

    cv2.destroyAllWindows()

def save_point_cloud(filename, points, colors):
    """
    Save 3D points and their corresponding colors to a PLY file.
    """
    points = points.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([points, colors])
    with open(filename, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {len(verts)}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('end_header\n')
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d')
    print(f"Point cloud saved to {filename}")

if __name__ == "__main__":
    main()
