import cv2

def initialize_camera(index, width=640, height=480, fps=30):
    """
    Initializes a camera given its index with specified resolution and frame rate.
    """
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    
    if not cap.isOpened():
        raise IOError(f"Camera {index} could not be opened.")
    
    return cap

def capture_frames(cap_left, cap_right):
    """
    Captures frames from two cameras.
    Note: This basic approach reads frames sequentially and may not be perfectly synchronized.
    For stricter synchronization, hardware triggers or multithreading may be required.
    """
    ret_left, frame_left = cap_left.read()
    ret_right, frame_right = cap_right.read()

    if not ret_left or not ret_right:
        raise RuntimeError("Failed to capture frames from both cameras.")
    
    return frame_left, frame_right

def main():
    try:
        cap_left = initialize_camera(1)
        cap_right = initialize_camera(2)
    except IOError as e:
        print(e)
        return

    print("Press 'q' to exit.")

    while True:
        try:
            frame_left, frame_right = capture_frames(cap_left, cap_right)
        except RuntimeError as e:
            print(e)
            break

        # Optionally, display frames side by side for visual inspection.
        combined_frame = cv2.hconcat([frame_left, frame_right])
        cv2.imshow("Stereo Image Acquisition", combined_frame)

        # Exit loop when 'q' is pressed.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
