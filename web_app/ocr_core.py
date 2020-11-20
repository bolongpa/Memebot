try:
    from PIL import Image
except ImportError:
    import Image
import cv2
import pytesseract
import sys


def ocr_core(filename):
    def meme_process(img):
        if img is None:
            return
    
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if all([elem >= 250 for elem in img[i][j]]):
                    img[i][j] = (0, 0, 0)
                else:
                    img[i][j] = (255, 255, 255)
        return img

    # Provided that tesseract is installed: https://tesseract-ocr.github.io/tessdoc/Compiling.html
    pytesseract.pytesseract.tesseract_cmd = r"/usr/local/bin/tesseract"
    image = cv2.imread(filename)
    image = meme_process(image)
    custom_config = r'-l eng --psm 1 -c tessedit_char_blacklist=1?|><.'

    result = pytesseract.image_to_string(image, config=custom_config)
    result = result.split()
    print(result)
    return result


if __name__ == "__main__":
    ocr_core(sys.argv[1])