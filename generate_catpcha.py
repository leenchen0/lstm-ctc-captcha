import os
from captcha_generator import CaptchaGenerator

# Captcha param
CHARSET = '0123456789'
CODE_MAX_LEN = 10
CODE_MIN_LEN = 4

FONT = './fonts/simsun.ttc'
FONT_SIZE = 28

CAPTCHA_SIZE = (230, 36) # (width, height)
CAPTCHA_NOISE_POINT = True
CAPTCHA_NOISE_LINE = True

# Dataset param
TRAIN_DATA_FOLDER = './train'
TEST_DATA_FOLDER = './test'

TRAIN_SIZE = 1000
TEST_SIZE = 100

DELETE_ALL_BEFORE_CREATE = True

def del_files(path, ext):
    for root, _, files in os.walk(path):
        for filename in files:
            if filename.endswith('.%s' % ext):
                os.remove(os.path.join(root, filename))

def main():
    generator = CaptchaGenerator(
        CHARSET, CODE_MIN_LEN, CODE_MAX_LEN,
        CAPTCHA_SIZE, FONT, FONT_SIZE,
        CAPTCHA_NOISE_POINT, CAPTCHA_NOISE_LINE
    )

    os.makedirs(TRAIN_DATA_FOLDER, exist_ok=True)
    os.makedirs(TEST_DATA_FOLDER, exist_ok=True)

    if DELETE_ALL_BEFORE_CREATE:
        del_files(TRAIN_DATA_FOLDER, 'png')
        del_files(TEST_DATA_FOLDER, 'png')

    generator.generate(TRAIN_DATA_FOLDER, TRAIN_SIZE)
    generator.generate(TEST_DATA_FOLDER, TEST_SIZE)

if __name__ == "__main__":
    main()
