import os
import random
import math

from PIL import Image, ImageDraw, ImageFont

class CaptchaGenerator:

    def __init__(self, charset, min_len, max_len, image_size, font, font_size, noise_point, noise_line):
        self.charset = charset
        self.max_len = max_len
        self.min_len = min_len
        self.image_size = image_size
        self.font = font
        self.font_size = font_size
        self.noise_point = noise_point
        self.noise_line = noise_line

    def generate(self, save_folder, num_captcha):
        os.makedirs(save_folder, exist_ok=True)

        for i in range(num_captcha):
            print('Generating %d/%d...' % (i + 1, num_captcha), end='')

            # Search for an ungenerated verification code
            generatedText = True
            while generatedText:
                text = self._random_text(self.charset, random.randint(self.min_len, self.max_len))
                filename = os.path.join(save_folder, '%s.png' % text)
                generatedText = os.path.exists(filename)
            print(text)

            img = self._draw(self.image_size, text, self.font, self.font_size)
            if self.noise_line:
                self._add_noise_line(img, 8)
            if self.noise_point:
                self._add_noise_point(img, 36)

            with open(filename, 'wb') as file:
                img.save(file, format='png')

    def _draw(self, canvas_size, text, font, font_size):
        (width, height) = canvas_size

        img = Image.new('RGB', canvas_size)
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
            draw.text([x, y], c, fill='white', font=font)

        return img

    def _add_noise_line(self, img, num_lines):
        draw = ImageDraw.Draw(img)
        for _ in range(num_lines):
            x1 = random.randint(0, img.width)
            x2 = random.randint(0, img.width)
            y1 = random.randint(0, img.height)
            y2 = random.randint(0, img.height)
            draw.line((x1, y1, x2, y2), fill=self._random_color())

    def _add_noise_point(self, img, num_points):
        draw = ImageDraw.Draw(img)
        for _ in range(num_points):
            draw.point([random.randint(0, img.width), random.randint(0, img.height)], fill=self._random_color())
            x = random.randint(0, img.width)
            y = random.randint(0, img.height)
            draw.arc((x, y, x + 4, y + 4), 0, 90, fill=self._random_color())

    def _random_color(self):
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    def _random_text(self, charset, num):
        arr = []
        for _ in range(num):
            arr.append(random.choice(charset))
        return ''.join(arr)
