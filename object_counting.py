import cv2
import urllib.request, urllib.error
import numpy as np
from scaler import scaleDownImage

# ORB and BFMatcher init
orb = cv2.ORB_create(nfeatures=500) # nfeatures - Number of features to find, default 500
bf = cv2.BFMatcher()

# Template
template = cv2.imread("template.jpg")
kp_template, des_template = orb.detectAndCompute(template, None)
template_kp_img = cv2.drawKeypoints(template, kp_template, None)

# Load image from server
image_url = 'http://127.0.0.1:8000/capture.jpg'
#image_url = 'http://esp32cam.local/download'
while True:
    try:
        img_resp = urllib.request.urlopen(image_url, timeout=10)
        break
    except urllib.error.URLError:
        print("Failed to load image from URL")
        exit()
imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
img = cv2.imdecode(imgnp, -1)
img2 = img.copy()
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Find and draw contours
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(cv2.GaussianBlur(gray, (11, 11), 0), 30, 150, 3)
dilated = cv2.dilate(canny, (1, 1), iterations=2)
(cnt, _) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(img, cnt, -1, (0, 255, 0), 2)

# Discard contours with areas under 900
threshold_area = 900
contours = [contour for contour in cnt if cv2.contourArea(contour) > threshold_area]
for i in cnt: print(cv2.contourArea(i))

objects_found = 0
good_match_factor = 0.75 # Lower - more strict, higher - less strict
threshold_matches = 40 # How many good matches are required for an object to be counted
for cont in contours:
    # Create an empty image the size of the original
    mask = np.zeros_like(img2)
    cv2.drawContours(mask, [cont], -1, (255, 255, 255), cv2.FILLED)

    # Apply closing operation
    # This is done because the contours sometimes have small gaps and thus can't be filled in
    kernel = np.ones((100, 100), np.uint8) # These numbers can be tweaked to fill smaller or larger gaps
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Get the current object from the image
    masked_image = cv2.bitwise_and(img2, mask)

    # Detect features
    kp_query, des_query = orb.detectAndCompute(masked_image, None)

    # Match features
    matches = bf.knnMatch(des_template, des_query, k=2)
    good_matches = [[m] for m, n in matches if m.distance < good_match_factor*n.distance]
    print(len(good_matches))

    # Count object if it has enough good matches
    if len(good_matches) > threshold_matches: objects_found += 1

    # Display features
    query_kp_img = cv2.drawKeypoints(masked_image, kp_query, None)
    features_compared = cv2.drawMatchesKnn(template, kp_template, masked_image, kp_query, good_matches, None, flags=2)

    # Display current object, its mask, and features
    mask = scaleDownImage(mask, 0.4)
    masked_image = scaleDownImage(masked_image, 0.4)
    features_compared = scaleDownImage(features_compared, 0.4)
    cv2.imshow("mask", mask)
    cv2.imshow("masked image", masked_image)
    cv2.imshow("features", features_compared)

    # Destroy those windows after any key press
    cv2.waitKey(0)
    cv2.destroyWindow("mask")
    cv2.destroyWindow("masked image")
    cv2.destroyWindow("features")


# Final output
print("Objects found:", objects_found)

# Display images
scale = 0.4
canny = scaleDownImage(canny, scale)
img = scaleDownImage(img, scale)
cv2.imshow("Contours", canny)
cv2.imshow("Source", img)

# Exit program
cv2.waitKey(0)
cv2.destroyAllWindows()

# Send sleep signal
sleep_url = "http://esp32cam.local/confirm"
try:
    urllib.request.urlopen(sleep_url, timeout=1)
except:
    pass

# Send data over LoRa

# Shutdown Pi
