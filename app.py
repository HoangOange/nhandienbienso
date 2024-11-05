import math
import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import Preprocess

ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 9
RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

n = 1
Min_char = 0.01
Max_char = 0.09

def process_image(image_path):
    global n
    img = cv2.imread(image_path)
    original_img = img.copy()

    # Initial resizing for very large images
    max_dimension = 1080
    if img.shape[0] > max_dimension or img.shape[1] > max_dimension:
        scaling_factor = max_dimension / float(max(img.shape))
        img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor)

    # Load KNN model
    npaClassifications = np.loadtxt("classifications.txt", np.float32)
    npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)
    npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))
    kNearest = cv2.ml.KNearest_create()
    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

    # Image Preprocessing
    imgGrayscaleplate, imgThreshplate = Preprocess.preprocess(img)
    canny_image = cv2.Canny(imgThreshplate, 250, 255)
    kernel = np.ones((3, 3), np.uint8)
    dilated_image = cv2.dilate(canny_image, kernel, iterations=1)

    # Draw contour and filter out license plates
    contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    screenCnts = []

    # Try to find 4-point contours
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.06 * peri, True)
        if len(approx) == 4:
            screenCnts.append(approx)

    detected_plates = []
    for screenCnt in screenCnts:
        cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)

        # Find angle of the license plate
        (x1, y1) = screenCnt[0, 0]
        (x2, y2) = screenCnt[1, 0]
        (x3, y3) = screenCnt[2, 0]
        (x4, y4) = screenCnt[3, 0]
        array = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        sorted_array = sorted(array, reverse=True, key=lambda x: x[1])
        (x1, y1) = sorted_array[0]
        (x2, y2) = sorted_array[1]
        doi = abs(y1 - y2)
        ke = abs(x1 - x2)
        angle = math.atan(doi / ke) * (180.0 / math.pi)

        # Crop and align the license plate
        mask = np.zeros(imgGrayscaleplate.shape, np.uint8)
        new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1)

        (x, y ) = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))

        roi = img[topx:bottomx, topy:bottomy]
        imgThresh = imgThreshplate[topx:bottomx, topy:bottomy]
        ptPlateCenter = (bottomx - topx) / 2, (bottomy - topy) / 2

        if x1 < x2:
            rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, -angle, 1.0)
        else:
            rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, angle, 1.0)

        roi = cv2.warpAffine(roi, rotationMatrix, (bottomy - topy, bottomx - topx ))
        imgThresh = cv2.warpAffine(imgThresh, rotationMatrix, (bottomy - topy, bottomx - topx))
        roi = cv2.resize(roi, (0, 0), fx=3, fy=3)
        imgThresh = cv2.resize(imgThresh, (0, 0), fx=3, fy=3)

        # Preprocessing and character segmentation
        kerel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thre_mor = cv2.morphologyEx(imgThresh, cv2.MORPH_DILATE, kerel3)
        cont, hier = cv2.findContours(thre_mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        char_x_ind = {}
        char_x = []
        height, width, _ = roi.shape
        roiarea = height * width

        for ind, cnt in enumerate(cont):
            (x, y, w, h) = cv2.boundingRect(cont[ind])
            ratiochar = w / h
            char_area = w * h

            if (Min_char * roiarea < char_area < Max_char * roiarea) and (0.25 < ratiochar < 0.7):
                if x in char_x:
                    x = x + 1
                char_x.append(x)
                char_x_ind[x] = ind

        # Character recognition
        char_x = sorted(char_x)
        first_line = ""
        second_line = ""

        for i in char_x:
            (x, y, w, h) = cv2.boundingRect(cont[char_x_ind[i]])
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

            imgROI = thre_mor[y:y + h, x:x + w]
            imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
            npaROIResized = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))

            npaROIResized = np.float32(npaROIResized)
            _, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k=3)
            strCurrentChar = str(chr(int(npaResults[0][0])))

            if (y < height / 3):
                first_line = first_line + strCurrentChar
            else:
                second_line = second_line + strCurrentChar

        if any(char.isdigit() for char in first_line + second_line):
            detected_plates.append(f"Biển số {n}: {first_line} {second_line}")
            n += 1

    # Display original image in the UI
    original_img_resized = cv2.resize(original_img, (640, 480))
    original_img_rgb = cv2.cvtColor(original_img_resized, cv2.COLOR_BGR2RGB)
    original_img_pil = Image.fromarray(original_img_rgb)
    original_img_tk = ImageTk.PhotoImage(original_img_pil)

    original_image_label.config(image=original_img_tk)
    original_image_label.image = original_img_tk

    # Display recognized image in the UI
    img_resized = cv2.resize(img, (640, 480))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(img_pil)

    recognized_image_label.config(image=img_tk)
    recognized_image_label.image = img_tk

    # Display all detected license plates in the result_label
    if detected_plates:
        result_text = "\n".join(detected_plates)
        result_label.config(text=result_text)
    else:
        result_label.config(text="Không tìm thấy biển số nào.")

def select_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        process_image(file_path)

# Create UI
root = Tk()
root.title("Chương Trình Nhận Diện Biển Số Nhóm 13")
root.configure(background="#DCF4FF")  # Set background color to blue

# Set window size and center
root.geometry("1400x900+{}+{}".format(
    int((root.winfo_screenwidth() - 1400) / 2),
    int((root.winfo_screenheight() - 900) / 2)
))

# Label at the top center
label_group = Label(root, text="NHẬN DIỆN BẢNG SỐ XE BẰNG THUẬT TOÁN K-NEAREST NEIGHBORS", font=("Inter", 24), fg ="#00567E" ,bg="#DCF4FF" )
label_group.pack(pady=10)
label_group = Label(root, text="Môn: Xử lý thị giác máy tính - GHVH: Trần Nguyên Bảo - Thực hiện: Nhóm 13", font=("Inter", 20), fg ="#017DB7",bg="#DCF4FF")
label_group.pack(pady=10)

# Frame for images
image_frame = Frame(root, bg="white")
image_frame.pack(pady=10)

# Labels for displaying images
original_image_label = Label(image_frame, text="Ảnh Gốc", font=("Inter", 18), fg ="#00567E" ,bg="#DCF4FF")
original_image_label.pack(side=LEFT, padx=10)

recognized_image_label = Label(image_frame, text="Ảnh sau khi nhận diện", font=("Inter", 18), fg ="#00567E" ,bg="#DCF4FF")
recognized_image_label.pack(side=RIGHT, padx=10)

# Label for displaying results
result_label = Label(root, text="", font=("Inter", 18), bg="#DCF4FF", fg="#00567E")
result_label.pack(pady=10)

# Button to select image
select_button = Button(root, text="Chọn ảnh biển số", command=select_image, bg="#96DEFF", fg="#DCF4FF")
select_button.pack(pady=20)

# Start UI loop
root.mainloop()