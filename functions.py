import numpy as np
import cv2
import re
numbers = re.compile(r'(\d+)')


def get_correspondence_points(matches, kp1, kp2):
    list_kp1 = []
    list_kp2 = []

    for mat in matches:
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt

        list_kp1.append((x1, y1))
        list_kp2.append((x2, y2))
    return list_kp1, list_kp2


def features_detection(img, img2):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img, None)  # detects keypoints on an image
    kp2, des2 = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)  # match detected keypoints on both images
    return matches[:50], kp1, kp2


def plot_points_on_image(image, points):
    font = cv2.FONT_HERSHEY_SIMPLEX
    if type(points[0]) == tuple:
        for i in range(len(points)):
            image = cv2.circle(image, (int(points[i][0]), int(points[i][1])), 2, (0, 0, 255), 3)
            image = cv2.putText(image, str(i), (int(points[i][0]), int(points[i][1])), font, 0.5, (0, 0, 255), 1)
    else:
        image = cv2.circle(image, (int(points[0]), int(points[1])), 2, (0, 0, 255), 3)
        image = cv2.putText(image, 'Point', (int(points[0]), int(points[1])), font, 0.5, (0, 0, 255), 1)

    return image


def get_image_coordinates_xy(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    xy_coords = np.asmatrix(np.ones((img_gray.shape[0] * img_gray.shape[1], 3), np.int16))
    xy_coords[:, 0:2] = np.flip(np.column_stack(np.where(img_gray >= 0)), axis=1)
    return xy_coords


def get_p_dash(H, p):
    p_ = np.matmul(H, np.transpose(p))
    p_ = np.matrix(np.transpose(p_))
    return np.divide(p_, p_[:, 2]).astype(int)


def remove(type, val, matrix, del_from):
    if type == 'G':
        y = np.argwhere(matrix > val)
    elif type == 'L':
        y = np.argwhere(matrix < val)
    elif type == 'GE':
        y = np.argwhere(matrix >= val)
    elif type == 'LE':
        y = np.argwhere(matrix <= val)
    else:
        y = np.argwhere(matrix == val)

    for i in range(len(del_from)):
        del_from[i] = np.delete(del_from[i], y[:, 0], 0)
    return del_from


def forward_warping(img, H, output):
    xy_coords = get_image_coordinates_xy(img)
    p_ = get_p_dash(H, xy_coords)
    output[p_[:, 1], p_[:, 0], :] = img[xy_coords[:, 1], xy_coords[:, 0]]
    return output


def bilinear_interpolation(x, y, img):
    h, w = img.shape[0:2]
    upperxweight = 1 - ((np.ceil(x)) - x)
    lowerxweight = ((np.ceil(x)) - x)
    upperyweight = 1 - ((np.ceil(y)) - y)
    loweryweight = ((np.ceil(y)) - y)

    ll = np.multiply(lowerxweight, loweryweight)

    lu = np.multiply(lowerxweight, upperyweight)
    ul = np.multiply(upperxweight, loweryweight)
    uu = np.multiply(upperxweight, upperyweight)

    lowery = np.fmin(h - 1, (np.floor(y))).astype('int16')
    uppery = np.fmin(h - 1, (np.ceil(y))).astype('int16')
    lowerx = np.fmin(w - 1, (np.floor(x))).astype('int16')
    upperx = np.fmin(w - 1, (np.ceil(x))).astype('int16')

    l = np.squeeze(img[lowery, lowerx])
    u = np.squeeze(img[uppery, upperx])
    xx = np.multiply(ll, l) + np.multiply(lu, u) + np.multiply(ul, l) + np.multiply(uu, u)
    xx = np.resize(xx, (xx.shape[0], 1, 3))
    return xx

def check_H(H, img1_coor, img1, img2):
    img1 = img1.copy()
    img2 = img2.copy()
    img1 = plot_points_on_image(img1, [img1_coor[0], img1_coor[1]])
    p_ = np.dot(H, [img1_coor[0], img1_coor[1], 1])
    p_ = np.divide(p_, p_[2])
    img2 = plot_points_on_image(img2, [p_[0], p_[1]])
    temp1 = cv2.resize(img1, (500, 500), interpolation=cv2.INTER_AREA)
    temp2 = cv2.resize(img2, (500, 500), interpolation=cv2.INTER_AREA)
    cv2.imshow("img1", temp1)
    cv2.imshow("img2", temp2)
    cv2.waitKey(0)
def inverse_warping(output_img, projected_img, H):
    h, w = projected_img.shape[0:2]
    Hinv = np.linalg.inv(H)
    xy_coords = get_image_coordinates_xy(output_img)
    p_ = get_p_dash(Hinv, xy_coords)
    p_, xy_coords = remove('L', 0, p_, [p_, xy_coords])
    p_, xy_coords = remove('GE', w, p_[:, 0], [p_, xy_coords])
    p_, xy_coords = remove('GE', h, p_[:, 1], [p_, xy_coords])
    output_img[xy_coords[:, 1], xy_coords[:, 0], :] = bilinear_interpolation(p_[:, 0], p_[:, 1], projected_img)
    return output_img


def get_final_img(img):
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(grayscale, 0, 255, cv2.THRESH_OTSU)
    # cv2.imshow("thresholded", thresholded)
    # cv2.waitKey(0)
    bbox = cv2.boundingRect(thresholded)
    x, y, w, h = bbox
    img = img[y:y + h, x:x + w]
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    cv2.imwrite("out.jpg", img)


def get_min_max(corners):
    xmin = np.floor(corners[:, 0, 0].min()).astype('int')
    ymin = np.floor(corners[:, 0, 1].min()).astype('int')
    xmax = np.ceil(corners[:, 0, 0].max()).astype('int')
    ymax = np.ceil(corners[:, 0, 1].max()).astype('int')
    return xmin, ymin, xmax, ymax


def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

