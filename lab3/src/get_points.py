import cv2

def get_points(image):
    points = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            print(f"Point sélectionné: ({x}, {y})")

            # Optionnel: afficher le point sur l'image
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Image", image)

    cv2.imshow("Image", image)
    cv2.setMouseCallback("Image", mouse_callback)

    print("Clique sur l'image (ESC pour terminer)...")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC pour quitter
            break

    cv2.destroyAllWindows()
    return points

if __name__ == "__main__":
    img = cv2.imread("/Users/antoineguenette/Desktop/Scolaire/Baccalauréat/Programmation/SH26/GBM6710-labs/lab3/images/calib_imgs/imageL.png")
    pts = get_points(img)

    print("Points sélectionnés :", pts)