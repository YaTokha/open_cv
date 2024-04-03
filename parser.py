import cv2
from cv2 import cvtColor, COLOR_BGR2GRAY, imread, THRESH_BINARY, threshold, equalizeHist
from easyocr import Reader

def optimize_image_for_ocr(image_path):
    image = imread(image_path)
    gray_image = cvtColor(image, COLOR_BGR2GRAY)
    equalized_image = equalizeHist(gray_image)
    _, optimized_image = threshold(equalized_image, 128, 255, THRESH_BINARY + THRESH_BINARY)
    optimized_image_path = "optimized_" + image_path
    cv2.imwrite(optimized_image_path, optimized_image)
    return optimized_image_path

def extract_text_from_image(path_to_image, output_file='extracted_text.txt'):
    optimized_image_path = optimize_image_for_ocr(path_to_image)
    ocr_reader = Reader(['ru'], gpu=False)  # Использование GPU если доступно
    text_blocks = ocr_reader.readtext(optimized_image_path, detail=0, paragraph=True)

    try:
        with open(output_file, mode="w", encoding="utf-8") as outfile:
            for text_block in text_blocks:
                outfile.write("{}\n\n".format(text_block))
        return f"Результат сохранен в {output_file}"
    except Exception as e:
        return f"Ошибка при записи файла: {e}"

def execute_ocr():
    image_path = 'optimized_img.png'
    result_message = extract_text_from_image(image_path, 'final_output.txt')
    print(result_message)


if __name__ == '__main__':
    execute_ocr()
