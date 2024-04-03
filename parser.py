import easyocr


def textrec(filepath, text_file_name='result.txt'):
    reader = easyocr.Reader(['ru'])
    result = reader.readtext(filepath, detail=0, paragraph=True)

    with open(text_file_name, "w", newline='', encoding="utf-8") as file:
        for line in result:
            file.write(f'{line}\n\n')

    return f'result {text_file_name}'


def main():
    filepath = 'img.jpeg'
    print(textrec(filepath=filepath))


if __name__ == main():
    main()
