print("hello")
import os
import cv2
import numpy as np
import imghdr
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from skimage.feature import hog
from skimage.segmentation import find_boundaries
from sklearn.cluster import MeanShift, KMeans
#import matplotlib.pyplot as plt
i=0
def nmd(img,i):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    path = r"C:\Users\ASUS\Desktop\sia"
    filename = "x.png"
    cv2.imwrite(os.path.join(path, filename), hsv)
    # خوشه بندی رنگ ها با استفاده از KMeans
    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(hsv.reshape(-1, 3))
    segmented_image = kmeans.labels_.reshape(image.shape[:2])
    # یافتن مرزهای شی
    boundaries = find_boundaries(segmented_image)
    print(np.count_nonzero(boundaries))
    min_x=None
    max_x=None
    min_y = None
    max_y = None
    for t, row in enumerate(boundaries):
        if any(element !=0 for element in row):
            if(min_x is None):
                min_x=t
            max_x=t
    for t, col in enumerate(boundaries):
        if any(element !=0 for element in col):
            if(min_y is None):
                min_y=t
            max_y=t
    if min_x is None:
        min_x=0
    if max_x is None:
        max_x=boundaries.shape[0]
    if min_y is None:
        min_y = 0
    if max_y is None:
        max_y=boundaries.shape[1]

    # برش تصویر برای شامل شدن فقط شی
    cropped_image = image[min_y:max_y, min_x:max_x]
    #path = r"C:\Users\ASUS\Desktop\Nmd"
    #filename = f"{i}.png"
    cv2.imwrite(os.path.join(path, filename), cropped_image)
    return cropped_image


def lineArt(img_gray, i):
    img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
    kernel=np.ones((5,5),np.uint8)
    img_dilated=cv2.dilate(img_gray,kernel,iterations=10)
    img_diff=cv2.absdiff(img_dilated,img_gray)
    contour=255-img_diff
    #img_gray=cv2.cvtColor(img_gray,cv2.COLOR_GRAY2BGR)
    path=r"C:\Users\ASUS\Desktop\lineArt"
    filename=f"{i}.png"
    cv2.imwrite(os.path.join(path,filename),contour)
    return contour
def canny_img(img_gray, i):
    img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
    sketch=cv2.Canny(img_gray, threshold1=30,threshold2=255)
    path = r"C:\Users\ASUS\Desktop\lineArt2"
    filename = f"{i}.png"
    cv2.imwrite(os.path.join(path, filename), sketch)
    return sketch
def hog_img(img_gray, i):
    img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
    feature,image=hog(img_gray,orientations=9,pixels_per_cell=(16,16),cells_per_block=(2,2), visualize=True)
    path = r"C:\Users\ASUS\Desktop\hog"
    filename = f"{i}.png"
    cv2.imwrite(os.path.join(path, filename),image)
    return feature
def laplace_img(img_gray, i):
    img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
    kernel=np.array([[0,1,0],[1,-4,1],[0,1,0]])
    #x=lineArt(img_gray,i)
    x = np.uint8(np.absolute(cv2.filter2D(img_gray, -1,kernel)))
    path = r"C:\Users\ASUS\Desktop\laplacian"
    filename = f"{i}.png"
    cv2.imwrite(os.path.join(path, filename), x)
    return img_gray
def momentom(img,i):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    moments = cv2.moments(img)
    hu_moments = cv2.HuMoments(moments)
    return hu_moments
def hist_img(img,i)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# محاسبه هیستوگرام رنگ
    color_hist = cv2.calcHist([hsv_image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
# مسیر فولدر تصاویر
data_dir = 'C:\\Users\\ASUS\\Desktop\\leaves'

# اندازه تصاویر
image_size = (128, 128)

# لیست برای نگهداری داده‌ها و برچسب‌ها
data = []
labels = []
i =0
# خواندن و پیش‌پردازش تصاویر از فولدرها
for folder_name in os.listdir(data_dir):
    folder_path = os.path.join(data_dir, folder_name)
    if os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if imghdr.what(file_path):  # بررسی اینکه فایل یک تصویر است
                image = cv2.imread(file_path, cv2.IMREAD_COLOR)
                if image is not None:
                    #image = nmd(image, i)
                    i=i+1
                    print(i)
                    image2 = lineArt(image, i)
                    image3 = canny_img(image, i)
                    image4 = hog_img(image, i)
                    image5 = momentom(image, i)
                    # تغییر اندازه‌ی تصویر
                    image = cv2.resize(image, image_size)
                    image2 = cv2.resize(image2, image_size)
                    image3 = cv2.resize(image3, image_size)
                    image4 = cv2.resize(image4, image_size)
                    image5 = cv2.resize(image5, image_size)

                    # نرمال‌سازی مقادیر پیکسل‌ها به محدوده [0, 1]
                    image = (image / 255.0).flatten()
                    image2 = (image2 / 255.0).flatten()
                    image3 = (image3 / 255.0).flatten()
                    image4 = (image4 / 255.0).flatten()
                    image5 = (image5 / 255.0).flatten()

                    #features=np.concatenate((image,image2, image3, image4, image5))
                    features=np.concatenate((image,image5))
                    # افزودن تصویر به داده‌ها
                    data.append(features)
                    # افزودن برچسب (label) به لیست برچسب‌ها
                    labels.append(folder_name)
#display_images(data[:10])
print("done")
# تبدیل لیست‌ها به آرایه‌های numpy
data = np.array(data)
labels = np.array(labels)

# تقسیم داده‌ها به مجموعه‌های آموزش و تست
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=32)

# تعریف مدل‌ها
models = {
    'SVM': SVC(kernel='linear', random_state=32),
    'SVM2': SVC(kernel='poly', degree=3, random_state=42),
    'SVM3': SVC(kernel='poly', degree=2, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=3),
    'KNN2': KNeighborsClassifier(n_neighbors=4),
    'Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}
print("here")
# آموزش مدل‌ها و محاسبه دقت
accuracies = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies[model_name] = accuracy
    print(f'{model_name} Accuracy: {accuracy * 100:.2f}%')

# مقایسه دقت مدل‌ها
best_model_name = max(accuracies, key=accuracies.get)
best_accuracy = accuracies[best_model_name]
print(f'\nBest Model: {best_model_name} with Accuracy: {best_accuracy * 100:.2f}%')