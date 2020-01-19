# LSTM-CTC-CAPTCHA

使用 LSTM+CTC 实现不定长验证码识别

## 运行环境

- python 3.6.7
- tensorflow 1.13.1
- numpy 1.16.2
- Pillow 5.4.1

## 运行

### 依赖安装

```shell
pip install -r requirements.txt
```

### 验证码生成

运行 `generate_captcha.py` 生成验证码(代码参考了[python生成图片验证码 - huangql517的博客 - CSDN博客](https://blog.csdn.net/huangql517/article/details/81508912))

可以通过修改相关变量配置生成验证码规则：

|变量|作用|
|--|--|
|CHARSET|验证码可以出现的字符集|
|CODE_MAX_LEN, CODE_MIN_LEN|设置验证码最大最小长度|
|CAPTCHA_SIZE|验证码尺寸(width, height)|
|FONT|字体文件, 项目文件夹 `fonts\` 下有几种字体|
|FONT_SIZE|字体大小|
|CAPTCHA_NOISE_POINT|是否加入点噪音|
|CAPTCHA_NOISE_LINE|是否加入线噪音|
|TRAIN_DATA_FOLDER|生成训练数据的保存文件夹|
|TEST_DATA_FOLDER|生成测试数据的保存文件夹|
|TRAIN_SIZE|训练集大小|
|TEST_SIZE|测试集大小|
|DELETE_ALL_BEFORE_CREATE|生成前是否删除原有文件夹下图片|

#### 验证码样例：

| ![captcha](https://github.com/PencilCl/lstm-ctc-captcha/blob/master/captcha/8925.png)| ![captcha](https://github.com/PencilCl/lstm-ctc-captcha/blob/master/captcha/672362.png) | ![captcha](https://github.com/PencilCl/lstm-ctc-captcha/blob/master/captcha/32847456.png) |
|--|--|--|
| ![captcha](https://github.com/PencilCl/lstm-ctc-captcha/blob/master/captcha/0170.png)| ![captcha](https://github.com/PencilCl/lstm-ctc-captcha/blob/master/captcha/93667.png) | ![captcha](https://github.com/PencilCl/lstm-ctc-captcha/blob/master/captcha/3285170.png) |

### 模型训练

运行 `train_model.py` 可以重新训练模型。`trained_model` 下的模型即使用该文件中的配置训练，训练验证码数量为 8k。

### 模型测试

`trained_model` 下包含一个训练好的模型，可以直接用来测试。随机生成 1000 张验证码识别精度为 99.3%。

运行 `test_model.py` 进行模型测试，需要设置 `DATA_FOLDER` 为验证码文件夹。

## 网络结构

模型定义在 `model.py` 文件中。包含三层 CNN，两层 LSTM，以及最后一层全连接层。

原图像大小为 36 * 230 * 3。

通过第一层卷积层 + 最大池化 => 18 * 115 * 48

通过第二层卷积层 + 最大池化 => 9 * 115 * 64

通过第三层卷积层 + 最大池化 => 5 * 58 * 128

将卷积层输出 reshape 为 (58, 5 * 128)，输入到两层 LSTM。

(**注：train_model.py 及 test_model.py 内的 MAX_TIMESTEPS 需要与 LSTM 的输入长度保持一致，即设置为 58**)

通过 LSTM 层 => 58 * 128

利用全连接层，将 128 维特征映射到类别数 => 58 * 11
