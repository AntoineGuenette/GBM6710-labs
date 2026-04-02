import cv2
import numpy as np
import os

def get_grid_points(img_path:str, nb_rows:int=5, nb_cols:int=5) -> np.ndarray:
    """
    Interactively select four corner points of a grid and generate an interpolated grid of points.

    Parameters:
        img_path (str): Path to the input image.
        nb_rows (int): Number of rows in the grid.
        nb_cols (int): Number of columns in the grid.

    Returns:
        grid (np.ndarray): Array of shape (nb_rows, nb_cols, 2) containing interpolated image points.
    """
    points = []
    point_index = 0
    instructions = [
        "Please select the (x_max, y_max) grid point",
        "Please select the (x_max, y_min) grid point",
        "Please select the (x_min, y_max) grid point",
        "Please select the (x_min, y_min) grid point",
    ]

    def draw_instruction(img):
        display = img.copy()
        if point_index < len(instructions):
            text = instructions[point_index]
        elif point_index == len(instructions):
            text = "All corners have been selected. Please press Enter to see all points."
        else:
            text = "Press ESC to quit."
        cv2.putText(display, text, (45, 45), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 0, 255), 2, cv2.LINE_AA)
        return display

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            nonlocal point_index
            nonlocal image
            # Add selected point
            points.append((x, y))
            point_index += 1

            # Display the selected point on the image
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Image", draw_instruction(image))
    
    # Load the image
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError(f"Image not found or unreadable: {img_path}")

    cv2.imshow("Image", draw_instruction(image))
    cv2.setMouseCallback("Image", mouse_callback)

    # Keep the window open until ESC is pressed
    while True:
        cv2.imshow("Image", draw_instruction(image))
        key = cv2.waitKey(1) & 0xFF
        if key == 13 and point_index == len(instructions):  # Enter key
            break
        if key == 27:  # ESC key
            break
    cv2.destroyAllWindows()
    
    # Ensure exactly 4 corner points were selected
    if len(points) != 4:
        raise ValueError("Exactly 4 points must be selected.")

    # Unpack points (order follows instructions)
    (x_max_y_max, x_max_y_min, x_min_y_max, x_min_y_min) = points

    # Convert to NumPy arrays for interpolation
    p1 = np.array(x_max_y_max)
    p2 = np.array(x_min_y_max)
    p3 = np.array(x_max_y_min)
    p4 = np.array(x_min_y_min)

    # Create interpolation grid
    grid = np.zeros((nb_rows, nb_cols, 2), dtype=float)

    for i in range(nb_rows):
        v = i / (nb_rows - 1) if nb_rows > 1 else 0
        for j in range(nb_cols):
            u = j / (nb_cols - 1) if nb_cols > 1 else 0

            point = (
                (1 - u) * (1 - v) * p1 +
                (1 - u) * v * p2 +
                u * (1 - v) * p3 +
                u * v * p4
            )

            grid[i, j] = point

    # Display interpolated grid points row by row
    display_img = image.copy()

    for i in range(nb_rows):
        for j in range(nb_cols):
            x, y = grid[i, j].astype(int)
            cv2.circle(display_img, (x, y), 3, (255, 120, 0), -1)
            cv2.imshow("Interpolated Grid", display_img)

            key = cv2.waitKey(25) & 0xFF
            if key == 27:
                break
        else:
            continue
        break

    # Final pause before closing
    cv2.waitKey(0)
    cv2.destroyWindow("Interpolated Grid")
    return grid

def get_ball_points(img_path:str) -> np.ndarray:
    """
    Interactively select ball center points from an image.

    Parameters:
        img_path (str): Path to the input image.

    Returns:
        points (np.ndarray): Array of shape (3, 2) containing selected image points.
    """
    points = []
    point_index = 0
    instructions = [
        "Please select the center of the (x_max, y_max) ball",
        "Please select the center of the (x_max, y_min) ball",
        "Please select the center of the (x_min, y_max) ball",
    ]

    def draw_instruction(img):
        display = img.copy()
        if point_index < len(instructions):
            text = instructions[point_index]
        else :
            text = "All points have been selected. Please press ESC to quit."
        cv2.putText(display, text, (45, 45), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 0, 255), 2, cv2.LINE_AA)
        return display

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            nonlocal point_index
            nonlocal image
            # Add selected point
            points.append((x, y))
            point_index += 1

            # Display the selected point on the image
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Image", draw_instruction(image))
    
    # Load the image
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError(f"Image not found or unreadable: {img_path}")

    cv2.imshow("Image", draw_instruction(image))
    cv2.setMouseCallback("Image", mouse_callback)

    # Keep the window open until ESC is pressed
    while True:
        cv2.imshow("Image", draw_instruction(image))
        key = cv2.waitKey(1) & 0xFF
        if key == 13 and point_index == len(instructions):  # Enter key
            break
        if key == 27:  # ESC key
            break
    cv2.destroyAllWindows()
    
    # Ensure exactly 3 points were selected
    if len(points) != 3:
        raise ValueError("Exactly 3 points must be selected.")

    return np.array(points)

def get_phantom_points(img_path:str) -> np.ndarray:
    """
    Interactively select phantom corner points from an image.

    Parameters:
        img_path (str): Path to the input image.

    Returns:
        points (np.ndarray): Array of shape (4, 2) containing selected image points.
    """
    points = []
    point_index = 0
    instructions = [
        "Please select the top back right corner",
        "Please select the top back left corner",
        "Please select the top front right corner",
        "Please select the top front left corner",
    ]

    def draw_instruction(img):
        display = img.copy()
        if point_index < len(instructions):
            text = instructions[point_index]
        else :
            text = "All points have been selected. Please press ESC to quit."
        cv2.putText(display, text, (45, 45), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 0, 255), 2, cv2.LINE_AA)
        return display

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            nonlocal point_index
            nonlocal image
            # Add selected point
            points.append((x, y))
            point_index += 1

            # Display the selected point on the image
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Image", draw_instruction(image))
    
    # Load the image
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError(f"Image not found or unreadable: {img_path}")

    cv2.imshow("Image", draw_instruction(image))
    cv2.setMouseCallback("Image", mouse_callback)

    # Keep the window open until ESC is pressed
    while True:
        cv2.imshow("Image", draw_instruction(image))
        key = cv2.waitKey(1) & 0xFF
        if key == 13 and point_index == len(instructions):  # Enter key
            break
        if key == 27:  # ESC key
            break
    cv2.destroyAllWindows()
    
    # Ensure exactly 4 points were selected
    if len(points) != 4:
        raise ValueError("Exactly 4 points must be selected.")

    return np.array(points)
