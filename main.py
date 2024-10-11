import cv2
import numpy as np
import subprocess
import easyocr
import re
import time
import hashlib

# ADB 路径
adb_path = r'C:\Program Files\Netease\MuMu Player 12\shell\adb.exe'

# ADB 设备地址
adb_device_address = '127.0.0.1:7555'


# 连接到指定的 ADB 设备
def connect_adb():
    subprocess.run([adb_path, 'connect', adb_device_address])
    print(f"Connected to {adb_device_address}.")


# 通过 adb 截图
def capture_screenshot():
    subprocess.run([adb_path, '-s', adb_device_address, 'exec-out', 'screencap', '-p'],
                   stdout=open('screen.png', 'wb'))
    print("Screenshot captured.")


# 模拟在屏幕上绘制符号或数字
def draw_answer(answer):
    # 指定绘制区域的坐标（根据您的设备调整）
    x_start = 300  # 起始 x 坐标
    y_start = 800  # 起始 y 坐标
    width = 200    # 绘制区域的宽度
    height = 200   # 绘制区域的高度

    if answer == '>':
        # 绘制 '>' 符号，使用两条线段
        # 第一条线：左下到中心
        x1 = x_start
        y1 = y_start + height
        x2 = x_start + width // 2
        y2 = y_start + height // 2
        subprocess.run([adb_path, '-s', adb_device_address, 'shell', 'input', 'swipe',
                        str(x1), str(y1), str(x2), str(y2), '100'])
        time.sleep(0.1)
        # 第二条线：左上到中心
        x1 = x_start
        y1 = y_start
        x2 = x_start + width // 2
        y2 = y_start + height // 2
        subprocess.run([adb_path, '-s', adb_device_address, 'shell', 'input', 'swipe',
                        str(x1), str(y1), str(x2), str(y2), '100'])
    elif answer == '<':
        # 绘制 '<' 符号，使用两条线段
        # 第一条线：右下到中心
        x1 = x_start + width
        y1 = y_start + height
        x2 = x_start + width // 2
        y2 = y_start + height // 2
        subprocess.run([adb_path, '-s', adb_device_address, 'shell', 'input', 'swipe',
                        str(x1), str(y1), str(x2), str(y2), '100'])
        time.sleep(0.1)
        # 第二条线：右上到中心
        x1 = x_start + width
        y1 = y_start
        x2 = x_start + width // 2
        y2 = y_start + height // 2
        subprocess.run([adb_path, '-s', adb_device_address, 'shell', 'input', 'swipe',
                        str(x1), str(y1), str(x2), str(y2), '100'])
    elif answer == '=' or answer == '2':
        # 绘制 '=' 符号，使用两条水平线
        x1 = x_start
        x2 = x_start + width
        y1 = y_start + height // 3
        y2 = y1
        subprocess.run([adb_path, '-s', adb_device_address, 'shell', 'input', 'swipe',
                        str(x1), str(y1), str(x2), str(y2), '100'])
        time.sleep(0.1)
        y1 = y_start + 2 * height // 3
        y2 = y1
        subprocess.run([adb_path, '-s', adb_device_address, 'shell', 'input', 'swipe',
                        str(x1), str(y1), str(x2), str(y2), '100'])
    elif answer == '正确' or answer == '错误':
        # 对于 '正确' 和 '错误'，可以使用 ADB 输入文字
        subprocess.run([adb_path, '-s', adb_device_address, 'shell', 'input', 'text', answer])
    else:
        print(f"Unsupported answer: {answer}")


# 计算文本的哈希值
def get_text_hash(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()


# 主流程
if __name__ == "__main__":
    # 第一步：连接到 ADB 设备
    connect_adb()

    # 初始化上一次的题目哈希值
    last_question_hash = ''

    # 初始化 OCR 读取器
    reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)

    while True:
        # 第二步：截取屏幕截图
        capture_screenshot()

        # 第三步：读取图像
        image = cv2.imread('screen.png')

        # 第四步：裁剪题目区域
        x1, y1, x2, y2 = 130, 340, 770, 470  # 题目区域坐标
        question_area = image[y1:y2, x1:x2]

        # 第五步：图像预处理
        gray = cv2.cvtColor(question_area, cv2.COLOR_BGR2GRAY)
        # 自适应阈值二值化
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 15, 3)

        # 第六步：使用 EasyOCR 进行文本识别
        result = reader.readtext(thresh)

        # 第七步：提取文本内容
        texts_with_positions = []
        for res in result:
            bbox, text, confidence = res
            # 过滤低置信度结果
            if confidence < 0.5:
                continue
            # 获取文字的平均 x 坐标
            x_coords = [point[0] for point in bbox]
            x_avg = sum(x_coords) / len(x_coords)
            texts_with_positions.append((x_avg, text))

        # 按照 x 坐标排序
        texts_with_positions.sort(key=lambda x: x[0])

        # 按顺序拼接识别的文字
        recognized_text_ordered = ''.join([text for x, text in texts_with_positions])
        print("Recognized Text Ordered:", recognized_text_ordered)

        # 计算当前题目的哈希值
        current_question_hash = get_text_hash(recognized_text_ordered)

        # 检测题目是否变化
        if current_question_hash == last_question_hash:
            print("Question has not changed. Waiting...")
            time.sleep(1)
            continue
        else:
            print("New question detected.")
            last_question_hash = current_question_hash

        # 第八步：处理题目，计算答案
        # 处理两种情况：数字比较和单位换算
        answer = None

        # 匹配单位换算，考虑单位在前或在后的情况
        pattern_conversion = r'(?:([\d,]+)(万|亿)|(万|亿)([\d,]+))\s*=\s*([\d,]+)'
        match_conversion = re.search(pattern_conversion, recognized_text_ordered)
        if match_conversion:
            if match_conversion.group(1) and match_conversion.group(2):
                # 数字在前，单位在后
                number = int(match_conversion.group(1).replace(',', ''))
                unit = match_conversion.group(2)
            elif match_conversion.group(3) and match_conversion.group(4):
                # 单位在前，数字在后
                unit = match_conversion.group(3)
                number = int(match_conversion.group(4).replace(',', ''))
            else:
                print("Failed to extract number and unit.")
                continue
            value = int(match_conversion.group(5).replace(',', ''))
            print(f"Extracted: {number}{unit} = {value}")
            # 进行单位换算
            if unit == '万':
                factor = 10000
            elif unit == '亿':
                factor = 100000000
            else:
                factor = 1
            # 检查换算是否正确
            expected_value = number * factor
            if expected_value == value:
                answer = '正确'
            else:
                answer = '错误'
            print(f"Conversion Result: {number}{unit} = {value}, Answer: {answer}")
        else:
            # 匹配数字比较，例如 "12345?67890"
            pattern_compare = r'(\d+)\s*(\D*)\s*(\d+)'
            match_compare = re.search(pattern_compare, recognized_text_ordered)
            if match_compare:
                num1 = match_compare.group(1)
                symbol = match_compare.group(2).strip()
                num2 = match_compare.group(3)
                print(f"Extracted: {num1} {symbol} {num2}")
                num1 = int(num1)
                num2 = int(num2)
                if num1 > num2:
                    answer = '>'
                elif num1 < num2:
                    answer = '<'
                else:
                    answer = '='
                print(f"Comparison Result: {num1} {answer} {num2}")
                # 如果答案是 '='，可能被识别为 '2'，所以将 '2' 也视为 '='
                if answer == '=':
                    answer = '2'
            else:
                print("Failed to parse the question.")
                continue  # 跳过，等待下一个循环

        # 第九步：在屏幕上绘制答案
        draw_answer(answer)

        # 等待一段时间，然后继续下一个循环
        time.sleep(2)
