import numpy as np
import imutils
import cv2

file_name="1.jpg"

cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)


def get_head_mask(img):
    """
    Получаем маску головы
    Вырезаем БГ
    :param img: Данное изображение
    :return:    Возвращаетм маску с вырезанным БГ
    """
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    faces = faceCascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))    #Находим лица
    if len(faces) != 0:
        x, y, w, h = faces[0]
        (x, y, w, h) = (x - 40, y - 100, w + 80, h + 200)
        rect1 = (x, y, w, h)
        cv2.grabCut(img, mask, rect1, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)     #Обрезаем БГ вокруг головы
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')  # Забираем маску с БГ

    return mask2

def is_bold(pnt, hair_mask):
    """
    Проверяем лысая ли голова
    :param pnt: Верхняя точка головы
    :param hair_mask: Маска с волосами
    :return: True если лысый, иначе False
    """
    roi = hair_mask[pnt[1]:pnt[1] + 40, pnt[0] - 40:pnt[0] + 40]    # Выделяем прямоугольник под верхней точкой
    cnt = cv2.countNonZero(roi) # Считаем кол-во ненулевых точек в этом прямоугольнике
    # Если кол-во точек меньше 25%, то считаем, что голова лысая
    if cnt < 800:
        print("Bold")
        return True
    else:
        print("Not Bold")
        return False


img1 = cv2.imread(file_name)     # Загружаем изображение
img1 = imutils.resize(img1, height=500)     # Приводим к 500px по высоте
mask = get_head_mask(img1)      # Получаем маску головы (без БГ)

# Находим контуры, берем самый большой и запомиаем его верхнюю точку как верхушку головы
cnts = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[1]
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
cnt=cnts[0]
topmost = tuple(cnt[cnt[:,:,1].argmin()][0])


# Убираем лицо по цвету кожи
lower = np.array([0, 0, 100], dtype="uint8")  # Нижний предел цвета кожи
upper = np.array([255, 255, 255], dtype="uint8")  # Верхний предел цвета кожи
converted = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)   # Переводим в цветовой формат HSV
skinMask = cv2.inRange(converted, lower, upper)     # Записываем маску из мест, где цвет находся между пределами
mask[skinMask == 255] = 0   # Убираем из маски головы маску лица

kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
mask = cv2.dilate(mask, kernel1, iterations=1)
i1 = cv2.bitwise_and(img1, img1, mask=mask)

# Если голова лысая, то выводим, что лысый и выводим координаты верхней точки головы
if is_bold(topmost,mask):
    cv2.rectangle(img1,topmost,topmost,(0,0,255),5)
    print(topmost)

# Иначе пишем, что не лысый и выводи координаты самого большого контура
else:
    cnts = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    cv2.drawContours(img1,[cnts[0]],-1,(0,0,255),2)
    for c in cnts[0]:
        print(c)

# Выводим изображение в цикле
while True:
    cv2.imshow("image1", img1)
    # Выход на Esc
    if cv2.waitKey(5) == 27:
        break

