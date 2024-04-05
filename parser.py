import cv2
from easyocr import Reader


def preprocess_image(image_path, scale_factor=2.0):

    img = cv2.imread(image_path)
    width = int(img.shape[1] * scale_factor)
    height = int(img.shape[0] * scale_factor)
    dim = (width, height)

    resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)

    gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_img = clahe.apply(gray)

    bin_img = cv2.adaptiveThreshold(contrast_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)

    preprocessed_path = 'preprocessed_' + image_path
    cv2.imwrite(preprocessed_path, bin_img)
    return preprocessed_path


def perform_ocr(preprocessed_path, reader):
    result = reader.readtext(preprocessed_path, detail=0, paragraph=True,
                             low_text=0.4,
                             text_threshold=0.7,
                             mag_ratio=1.5)
    return result


def save_result(result, output_path='result.txt'):
    with open(output_path, 'w', encoding='utf-8') as file:
        for line in result:
            file.write(line + '\n')

def main():
    image_path = 'img.png'
    reader = Reader(['ru'], gpu=False)
    # preprocessed_path = preprocess_image(image_path)
    result = perform_ocr(image_path, reader)
    save_result(result)


if __name__ == '__main__':
    main()
