import numpy as np
import cv2


def distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def find_nearest_coordinate(point, coordinates):
    # Tìm tọa độ có khoảng cách gần nhất
    x, y = point
    nearest_coord = min(coordinates, key=lambda coord: distance(x, y, coord[0], coord[1]))
    return nearest_coord


def centroid_in_polygon(centroid, polygon):
    # Chuyển đổi các điểm của đa giác thành kiểu numpy array
    polygon = np.array(polygon, np.int32)
    centroid = tuple(float(x) for x in centroid)
    # Sử dụng hàm pointPolygonTest để kiểm tra
    result = cv2.pointPolygonTest(polygon, centroid, False)
    if result >= 0:
        return True
    else:
        return False
    

def draw_line(event, x, y, flags, param):
    global line_start, line_end, drawing, line_drawn

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        line_start = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            line_end = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        line_end = (x, y)
        line_drawn = True


def non_max_suppression(cars_in_frame, overlapThresh):
    '''
    boxes: List các bounding box
    overlapThresh: Ngưỡng overlapping giữa các hình ảnh
    '''
    cars = np.array(cars_in_frame)

    # Khởi tạo list của index được lựa chọn
    pick = []

    # Lấy ra tọa độ của các bounding boxes
    x1 = cars[:,0]
    y1 = cars[:,1]
    x2 = cars[:,2]
    y2 = cars[:,3]

    # Tính toàn diện tích của các bounding boxes và sắp xếp chúng theo thứ tự từ bottom-right, chính là tọa độ theo y của bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # Khởi tạo một vòng while loop qua các index xuất hiện trong indexes
    while len(idxs) > 0:
    # Lấy ra index cuối cùng của list các indexes và thêm giá trị index vào danh sách các indexes được lựa chọn
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # Tìm cặp tọa độ lớn nhất (x, y) là điểm bắt đầu của bounding box và tọa độ nhỏ nhất (x, y) là điểm kết thúc của bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # Tính toán width và height của bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # Tính toán tỷ lệ diện tích overlap
        overlap = (w * h) / area[idxs[:last]]

        # Xóa index cuối cùng và index của bounding box mà tỷ lệ diện tích overlap > overlapThreshold
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
        # Trả ra list các index được lựa chọn
    return [cars_in_frame[i] for i in pick]

