# coding: utf-8

import os
import random
import math

from PIL import Image, ImageDraw, ImageFont

CHARSET = '0123456789'
CODE_MAX_LEN = 10
CODE_MIN_LEN = 4
CAPTCHA_SIZE = (230, 36) # (width, height)

FONT = './fonts/simsun.ttc'
FONT_SIZE = 28

CAPTCHA_NOISE_POINT = True
CAPTCHA_NOISE_LINE = True

NUM_CAPTCHA = 100
SAVE_FOLDER = './test'
CLEAR_BEFORE_CREATE = True

def get_random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def draw(canvas_size, text, font, font_size, noise_point, noise_line):
    (width, height) = canvas_size

    img = Image.new("RGB", canvas_size)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font=font, size=font_size)

    len_text = len(text)
    char_width = width / len_text
    offset_range_height = math.floor(max(0, height - font_size))
    offset_range_width = math.floor(max(0, char_width - font_size))
    for i in range(len_text):
        c = text[i]
        x = i * char_width + random.randint(0, offset_range_width)
        y = random.randint(0, offset_range_height)
        draw.text([x, y], c, fill="white", font=font)

    if noise_line:
        for _ in range(8):
            x1 = random.randint(0, width)
            x2 = random.randint(0, width)
            y1 = random.randint(0, height)
            y2 = random.randint(0, height)
            draw.line((x1, y1, x2, y2), fill=get_random_color())

    if noise_point:
        for _ in range(36):
            draw.point([random.randint(0, width), random.randint(0, height)], fill=get_random_color())
            x = random.randint(0, width)
            y = random.randint(0, height)
            draw.arc((x, y, x + 4, y + 4), 0, 90, fill=get_random_color())

    return img

def random_text(charset, num):
    arr = []
    for _ in range(num):
        arr.append(random.choice(charset))

    return ''.join(arr)

def del_files(path, ext):
    for root, _, files in os.walk(path):
        for filename in files:
            if filename.endswith('.%s' % ext):
                os.remove(os.path.join(root, filename))

def main():
    os.makedirs(SAVE_FOLDER, exist_ok=True)
    if CLEAR_BEFORE_CREATE:
        del_files(SAVE_FOLDER, 'png')

    for i in range(NUM_CAPTCHA):
        print('Generating %d/%d...' % (i + 1, NUM_CAPTCHA), end='')

        # Search for an ungenerated verification code
        generatedText = True
        while generatedText:
            text = random_text(CHARSET, random.randint(CODE_MIN_LEN, CODE_MAX_LEN))
            filename = os.path.join(SAVE_FOLDER, '%s.png' % text)
            generatedText = os.path.exists(filename)
        print(text)

        img = draw(
            CAPTCHA_SIZE, text,
            FONT, FONT_SIZE,
            CAPTCHA_NOISE_POINT, CAPTCHA_NOISE_LINE
        )

        with open(filename, "wb") as file:
            img.save(file, format="png")

if __name__ == "__main__":
    main()
