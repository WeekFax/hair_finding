import numpy as np
import imutils
import cv2


def get_skin_mask(img):
    """
    Получаем маску кожи
    :param img: Данное изображение
    :return:    Возвращает маску с участками кожи
    """
    lower = np.array([0, 48, 80], dtype = "uint8")  # Нижний предел цвета кожи
    upper = np.array([20, 255, 255], dtype = "uint8")   # Верхний предел цвета кожи
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Преобразовываем данное изображение к цветовой схеме HSV
    skinMask = cv2.inRange(converted, lower, upper) # Создаем маску с местами с цветами, между верхним и нижним пределами


    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skinMask = cv2.erode(skinMask, kernel1, iterations=2)
    skinMask = cv2.dilate(skinMask, kernel1, iterations=2)

    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23))
    skinMask = cv2.morphologyEx(skinMask, cv2.MORPH_OPEN, kernel2)
    skinMask = cv2.GaussianBlur(skinMask, (3,3), 0)
    return skinMask

def get_head_mask(img):
    """
    Получаем маску головы
    Вырезаем БГ
    :param img: Данное изображение
    :return:    Возвращаетм маску с вырезанным БГ
    """

    mask = np.zeros(img.shape[:2], np.uint8)    # Создаем пустую маску
    h, w = img.shape[:2]
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (20, 20, w - 40, h - 40)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)  # Создаем маски с БГ и ФГ
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')   # Забираем маску с БГ
    return mask2

def get_hair_mask(img):
    """
    Получаем маску волос
    :param img: Данное изображение
    :return: Возвращает маску волос
    """
    mask1 = get_head_mask(frame)    # Получаем маску головы
    mask2 = cv2.bitwise_not(get_skin_mask(frame))    # Получаем маску с кожей и инвертируем её

    mask = cv2.bitwise_and(mask1, mask2)    # Создаем маску из общего двух масок

    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel2)  # Удаляем лишние пятна в маске
    return mask


frame=cv2.imread("1.jpg")

frame = imutils.resize(frame, height=500)   # Задаем размеры изобрадения (фильтры настроены под это размер

hair_mask=get_hair_mask(frame)
hair=cv2.bitwise_and(frame, frame, mask=hair_mask)  # Вырезаем волосы по маске

# Находим контуры на маске
cnts = cv2.findContours(hair_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

# Выводим координаты контура
for c in cnts:
    for b in c:
        print(b)
    #cv2.drawContours(frame, [c], -1, (255, 0, 0), 1)

# Объединяем изображения до и после
frame=np.hstack([frame, hair])
frame = imutils.resize(frame, height=500)

while True:
    cv2.imshow("images",frame)
    if cv2.waitKey(5)==27:
        break