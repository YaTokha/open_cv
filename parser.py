import cv2
import numpy as np
from easyocr import Reader


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_img = clahe.apply(gray)
    _, bin_img = cv2.threshold(contrast_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    preprocessed_path = 'preprocessed_' + image_path
    cv2.imwrite(preprocessed_path, bin_img)
    return preprocessed_path


def perform_ocr(preprocessed_path, reader):
    result = reader.readtext(preprocessed_path, detail=0, paragraph=True)
    return result


def save_result(result, output_path='result.txt'):
    with open(output_path, 'w', encoding='utf-8') as file:
        for line in result:
            file.write(line + '\n')


def main():
    image_path = 'preprocessed_img.png'
    reader = Reader(['ru'], gpu=False)  # Выключаем GPU для совместимости
    preprocessed_path = preprocess_image(image_path)
    result = perform_ocr(preprocessed_path, reader)
    save_result(result)


if __name__ == '__main__':
    main()
