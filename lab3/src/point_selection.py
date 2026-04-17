import cv2
import numpy as np

def generate_grid_world(x_max:float, y_max:float, spacing:float=25.0, n:int=5):
    """
    Generate a grid of world points ordered from (x_max, y_max) to (x_min, y_min), matching the
    expected calibration ordering.

    Parameters:
        x_max (float): X coordinate of the top-right corner
        y_max (float): Y coordinate of the top-right corner
        spacing (float): Distance between adjacent points (default: 25 mm)
        n (int): number of points per row/column (default: 5)

    Returns:
        np.ndarray: (n, n, 3) array of world points
    """
    pts = []

    # Loop over rows (y direction)
    for i in range(n): 
        row = []
        # Loop over columns (x direction)
        for j in range(n):
            x = x_max - i * spacing
            y = y_max - j * spacing
            row.append([x, y, 0.0])
        pts.append(row)
    # Convert list of points to NumPy array
    return np.array(pts)

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

    # Display instructions on the image
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

    # Handle mouse clicks to record selected points
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
    
    # Load input image
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError(f"Image not found or unreadable: {img_path}")
    cv2.imshow("Image", draw_instruction(image))
    cv2.setMouseCallback("Image", mouse_callback)

    # Wait for user to finish selecting points
    while True:
        cv2.imshow("Image", draw_instruction(image))
        key = cv2.waitKey(1) & 0xFF
        if key == 13 and point_index == len(instructions):  # Enter key
            break
        if key == 27:  # ESC key
            break
    cv2.destroyAllWindows()
    
    # Validate number of selected points
    if len(points) != 4:
        raise ValueError("Exactly 4 points must be selected.")

    # Assign selected corner points
    (x_max_y_max, x_max_y_min, x_min_y_max, x_min_y_min) = points

    # Convert points to NumPy format
    p1 = np.array(x_max_y_max)
    p2 = np.array(x_min_y_max)
    p3 = np.array(x_max_y_min)
    p4 = np.array(x_min_y_min)

    # Initialize interpolated grid
    grid = np.zeros((nb_rows, nb_cols, 2), dtype=float)

    # Iterate over grid rows
    for i in range(nb_rows):
        v = i / (nb_rows - 1) if nb_rows > 1 else 0
        # Iterate over grid columns
        for j in range(nb_cols):
            u = j / (nb_cols - 1) if nb_cols > 1 else 0

            # Compute interpolated point using bilinear interpolation
            point = (
                (1 - u) * (1 - v) * p1 +
                (1 - u) * v * p2 +
                u * (1 - v) * p3 +
                u * v * p4
            )
            grid[i, j] = point

    # Visualize interpolated grid points
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

    # Wait before closing visualization window
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

    # Display instructions on the image
    def draw_instruction(img):
        display = img.copy()
        if point_index < len(instructions):
            text = instructions[point_index]
        else :
            text = "All points have been selected. Please press ESC to quit."
        cv2.putText(display, text, (45, 45), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 0, 255), 2, cv2.LINE_AA)
        return display

    # Handle mouse clicks to record selected points
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
    
    # Load input image
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError(f"Image not found or unreadable: {img_path}")
    cv2.imshow("Image", draw_instruction(image))
    cv2.setMouseCallback("Image", mouse_callback)

    # Wait for user to finish selecting points
    while True:
        cv2.imshow("Image", draw_instruction(image))
        key = cv2.waitKey(1) & 0xFF
        if key == 13 and point_index == len(instructions):  # Enter key
            break
        if key == 27:  # ESC key
            break
    cv2.destroyAllWindows()
    
    # Validate number of selected points
    if len(points) != 3:
        raise ValueError("Exactly 3 points must be selected.")

    # Convert selected points to NumPy array
    points = np.array(points)

    # Prepare image for circle detection
    image_color = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)

    # Process each selected point
    detected_centers = []
    for (x, y) in points:
        x, y = int(x), int(y)

        # Extract local region around point
        roi_size = 50
        x_min = max(x - roi_size, 0)
        x_max = min(x + roi_size, gray.shape[1])
        y_min = max(y - roi_size, 0)
        y_max = min(y + roi_size, gray.shape[0])

        roi = gray[y_min:y_max, x_min:x_max]

        # Detect circles in the ROI
        circles = cv2.HoughCircles(
            roi,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=30,
            param1=100,
            param2=20,
            minRadius=10,
            maxRadius=100
        )

        # Check if any circles were detected
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")

            # Find closest detected circle to clicked point
            best_circle = None
            min_dist = float("inf")

            for (cx, cy, r) in circles:
                dist = np.sqrt((cx - (x - x_min))**2 + (cy - (y - y_min))**2)
                if dist < min_dist:
                    min_dist = dist
                    best_circle = (cx, cy, r)

            # Use best matching circle
            if best_circle is not None:
                cx, cy, r = best_circle

                # Convert coordinates back to full image
                cx_full = cx + x_min
                cy_full = cy + y_min

                detected_centers.append((cx_full, cy_full))

                # Draw circle contour and center
                cv2.circle(image_color, (cx_full, cy_full), r, (255, 120, 0), 2)
                cv2.circle(image_color, (cx_full, cy_full), 4, (255, 120, 0), -1)
        else:
            # Fallback if no circle is detected
            detected_centers.append((x, y))
            cv2.circle(image_color, (x, y), 4, (255, 0, 0), -1)

    # Display detected circle centers
    cv2.imshow("Detected Circles", image_color)
    cv2.waitKey(0)
    cv2.destroyWindow("Detected Circles")

    return np.array(detected_centers)

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

    # Display instructions on the image
    def draw_instruction(img):
        display = img.copy()
        if point_index < len(instructions):
            text = instructions[point_index]
        else :
            text = "All points have been selected. Please press ESC to quit."
        cv2.putText(display, text, (45, 45), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 0, 255), 2, cv2.LINE_AA)
        return display

    # Handle mouse clicks to record selected points
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
    
    # Load input image from disk
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError(f"Image not found or unreadable: {img_path}")
    cv2.imshow("Image", draw_instruction(image))
    cv2.setMouseCallback("Image", mouse_callback)

    # Wait for user to finish selecting points
    while True:
        cv2.imshow("Image", draw_instruction(image))
        key = cv2.waitKey(1) & 0xFF
        if key == 13 and point_index == len(instructions):  # Enter key
            break
        if key == 27:  # ESC key
            break
    cv2.destroyAllWindows()
    
    # Validate number of selected points
    if len(points) != 4:
        raise ValueError("Exactly 4 points must be selected.")

    return np.array(points)
