import cv2
import os

def create_folders():
    """
    Create folders to store left and right images if they do not exist.
    """
    if not os.path.exists("left"):
        os.makedirs("left")
    if not os.path.exists("right"):
        os.makedirs("right")

def main():

    create_folders()

    cap_left = cv2.VideoCapture(0)
    cap_right = cv2.VideoCapture(2)

    if not cap_left.isOpened() or not cap_right.isOpened():
        print("Error: Could not open one or both cameras.")
        return

    # cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    capture_index = 0  

    print("Press 'c' to capture image pair, or 'q' to quit.")

    while True:
        ret_left, frame_left = cap_left.read()
        ret_right, frame_right = cap_right.read()

        if not ret_left or not ret_right:
            print("Error: Failed to capture frame(s).")
            break


        cv2.imshow("Left Camera", frame_left)
        cv2.imshow("Right Camera", frame_right)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):

            left_filename = f"left/left_{capture_index:03d}.jpg"
            right_filename = f"right/right_{capture_index:03d}.jpg"
            cv2.imwrite(left_filename, frame_left)
            cv2.imwrite(right_filename, frame_right)
            print(f"Captured image pair {capture_index}")
            capture_index += 1

        elif key == ord('q'):

            break

    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
