from paddleocr import PaddleOCR

# Force old stable behavior
ocr = PaddleOCR(
    lang='en',
    use_angle_cls=True,
    show_log=False
)

def extract_text_from_image(image_path: str):
    result = ocr.ocr(image_path)

    extracted_text = ""
    for line in result:
        for word in line:
            extracted_text += word[1][0] + " "

    return extracted_text.strip()