import cv2
import numpy as np
import subprocess
import threading
import time
import pytesseract
import os

# 获取脚本所在目录，并构建 res/ 目录的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(script_dir, 'res')

# ADB 路径
adb_path = r'C:\Program Files\Netease\MuMu Player 12\shell\adb.exe'  # 请根据实际路径调整

# 手动指定 tesseract.exe 的路径
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # 请根据实际路径调整

# 全局调试图像
debug_image = None


# 通过 adb 获取截屏到内存中
def capture_screenshot():
    result = subprocess.run([adb_path, 'exec-out', 'screencap', '-p'], stdout=subprocess.PIPE)
    screenshot_bytes = result.stdout

    # 将截屏数据转换为 numpy 数组并解码为图像
    image_array = np.frombuffer(screenshot_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise RuntimeError("Failed to capture screenshot.")

    return image


# 使用 ORB 特征匹配检测按钮位置
def match_template(image, template_filename):
    global debug_image
    print(f"Matching template: {template_filename}")
    template_full_path = os.path.join(template_dir, template_filename)
    template = cv2.imread(template_full_path, cv2.IMREAD_GRAYSCALE)

    if template is None:
        print(f"Failed to load template image: {template_full_path}")
        return None

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 限定搜索区域（根据实际按钮位置调整）
    roi_x1, roi_y1, roi_x2, roi_y2 = 100, 100, 1000, 800  # 示例值，需要根据实际情况调整
    roi = gray_image[roi_y1:roi_y2, roi_x1:roi_x2]
    gray_image_roi = roi.copy()

    # 初始化 ORB 检测器
    orb = cv2.ORB_create(nfeatures=500)
    # 找到关键点和描述符
    kp1, des1 = orb.detectAndCompute(template, None)
    kp2, des2 = orb.detectAndCompute(gray_image_roi, None)

    if des1 is None or des2 is None:
        print("No descriptors found, cannot match.")
        return None

    # 创建 BFMatcher 对象
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    # knnMatch 匹配
    matches = bf.knnMatch(des1, des2, k=2)

    # 应用比值测试（Lowe's ratio test）
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    print(f"Number of good matches: {len(good_matches)}")

    # 调整最小匹配数量阈值
    MIN_MATCH_COUNT = 15
    if len(good_matches) >= MIN_MATCH_COUNT:
        # 获取匹配的关键点坐标
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # 计算变换矩阵
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if M is not None:
            h, w = template.shape
            # 使用变换矩阵将模板的角点映射到待匹配图像中
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

            # 计算匹配区域的面积
            area = cv2.contourArea(dst)
            template_area = h * w
            area_ratio = area / template_area

            # 验证面积比例是否合理
            if 0.5 < area_ratio < 2.0:
                # 计算中心点（加上 ROI 的偏移）
                center_x = int(np.mean(dst[:, 0, 0])) + roi_x1
                center_y = int(np.mean(dst[:, 0, 1])) + roi_y1
                print(f"Matched Coordinates: ({center_x}, {center_y})")

                # 画出匹配区域
                image_matched = image.copy()
                dst += np.array([roi_x1, roi_y1], dtype=np.float32)
                cv2.polylines(image_matched, [np.int32(dst)], True, (0, 255, 0), 3)

                # 更新调试图像
                debug_image = image_matched.copy()

                return center_x, center_y
            else:
                print("Area ratio not within expected range.")
                return None
        else:
            print("Homography could not be computed.")
            return None
    else:
        print("Not enough good matches, no match found.")
        return None


# 使用 adb 在作答区域划线绘制结果
def draw_result_on_screen(result, coords):
    global debug_image
    if result is None:
        print("无法识别足够的数字来进行比较")
        return

    x1, y1, x2, y2 = coords
    middle_x = (x1 + x2) // 2
    middle_y = (y1 + y2) // 2
    offset = 50  # 定义偏移量，用于绘制符号的线条位置

    # 时间参数用于加快 swipe 持续时间（单位：毫秒）
    swipe_time = 50  # 将滑动时间设为合理的滑动时间

    # 根据比较结果绘制符号
    if result == ">":
        # 绘制 '>' 符号（右向箭头）
        subprocess.run([adb_path, 'shell', 'input', 'swipe', str(middle_x - offset), str(middle_y - offset),
                        str(middle_x + offset), str(middle_y), str(swipe_time)])
        subprocess.run([adb_path, 'shell', 'input', 'swipe', str(middle_x - offset), str(middle_y + offset),
                        str(middle_x + offset), str(middle_y), str(swipe_time)])
        # 在调试图像上绘制结果
        cv2.line(debug_image, (middle_x - offset, middle_y - offset), (middle_x + offset, middle_y), (0, 0, 255), 5)
        cv2.line(debug_image, (middle_x - offset, middle_y + offset), (middle_x + offset, middle_y), (0, 0, 255), 5)
    elif result == "<":
        # 绘制 '<' 符号（左向箭头）
        subprocess.run([adb_path, 'shell', 'input', 'swipe', str(middle_x + offset), str(middle_y - offset),
                        str(middle_x - offset), str(middle_y), str(swipe_time)])
        subprocess.run([adb_path, 'shell', 'input', 'swipe', str(middle_x + offset), str(middle_y + offset),
                        str(middle_x - offset), str(middle_y), str(swipe_time)])
        # 在调试图像上绘制结果
        cv2.line(debug_image, (middle_x + offset, middle_y - offset), (middle_x - offset, middle_y), (0, 0, 255), 5)
        cv2.line(debug_image, (middle_x + offset, middle_y + offset), (middle_x - offset, middle_y), (0, 0, 255), 5)
    elif result == "=":
        # 绘制 '=' 符号（两条水平线）
        subprocess.run([adb_path, 'shell', 'input', 'swipe', str(middle_x - offset), str(middle_y - 20),
                        str(middle_x + offset), str(middle_y - 20), str(swipe_time)])
        subprocess.run([adb_path, 'shell', 'input', 'swipe', str(middle_x - offset), str(middle_y + 20),
                        str(middle_x + offset), str(middle_y + 20), str(swipe_time)])
        # 在调试图像上绘制结果
        cv2.line(debug_image, (middle_x - offset, middle_y - 20), (middle_x + offset, middle_y - 20), (0, 0, 255), 5)
        cv2.line(debug_image, (middle_x - offset, middle_y + 20), (middle_x + offset, middle_y + 20), (0, 0, 255), 5)

    # 显示结果图像
    cv2.imshow('Debug Window', debug_image)
    cv2.waitKey(1)

    # 打印到控制台
    print(f"Drawing result '{result}' using swipes at coordinates near ({middle_x}, {middle_y})")


# 识别题目区域内的两个数字
def recognize_two_numbers(image):
    global debug_image
    # 定义两个数字区域的坐标
    first_num_coords = (330, 250, 740, 350)  # (x1, y1, x2, y2)
    second_num_coords = (850, 250, 1200, 350)

    # 裁剪数字区域
    first_num_image = image[first_num_coords[1]:first_num_coords[3], first_num_coords[0]:first_num_coords[2]].copy()
    second_num_image = image[second_num_coords[1]:second_num_coords[3],
                       second_num_coords[0]:second_num_coords[2]].copy()

    # 使用 Tesseract 识别数字
    first_number = recognize_number_with_tesseract(first_num_image, "First Number")
    second_number = recognize_number_with_tesseract(second_num_image, "Second Number")

    print(f"Recognized numbers: {first_number}, {second_number}")

    # 在调试图像上绘制识别区域和结果
    debug_image = image.copy()
    cv2.rectangle(debug_image, (first_num_coords[0], first_num_coords[1]), (first_num_coords[2], first_num_coords[3]),
                  (255, 0, 0), 2)
    cv2.rectangle(debug_image, (second_num_coords[0], second_num_coords[1]),
                  (second_num_coords[2], second_num_coords[3]), (0, 255, 0), 2)

    # 显示识别结果
    if first_number is not None:
        cv2.putText(debug_image, f"{first_number}", (first_num_coords[0], first_num_coords[3] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    else:
        cv2.putText(debug_image, "N/A", (first_num_coords[0], first_num_coords[3] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if second_number is not None:
        cv2.putText(debug_image, f"{second_number}", (second_num_coords[0], second_num_coords[3] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(debug_image, "N/A", (second_num_coords[0], second_num_coords[3] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 显示调试窗口
    cv2.imshow('Debug Window', debug_image)
    cv2.waitKey(1)

    return first_number, second_number


# 使用 Tesseract 识别单个数字
def recognize_number_with_tesseract(image, position):
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 应用对比度拉伸增强图像
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)

    # 应用高斯模糊以减少噪点
    blurred = cv2.GaussianBlur(enhanced_gray, (5, 5), 0)

    # 应用自适应阈值
    binary_image = cv2.adaptiveThreshold(blurred, 255,
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 11, 2)

    # Tesseract 配置，仅识别数字
    custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
    text = pytesseract.image_to_string(binary_image, config=custom_config)

    # 增加调试信息
    print(f"OCR Result ({position}): {text}")

    # 仅保留数字字符
    cleaned_text = ''.join([char for char in text if char.isdigit()])

    # 提取连续数字
    numbers = ''.join(cleaned_text)

    recognized_number = int(numbers) if numbers else None

    # 在图像上绘制识别到的数字
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)

    return recognized_number


# 比较两个数值并返回符号
def compare_numbers(num1, num2):
    if num1 is None or num2 is None:
        return None  # 无法比较
    if num1 > num2:
        return ">"
    elif num1 < num2:
        return "<"
    else:
        return "="


# 使用多模板匹配，支持多个模板
def match_template_with_multiple_templates(image, template_filenames):
    for template_filename in template_filenames:
        coords = match_template(image, template_filename)
        if coords:
            return coords
    return None


# 检测并点击按钮
def check_and_click_buttons():
    global debug_image
    while True:
        try:
            image = capture_screenshot()
            # 多个“开心收下”模板
            happy_templates = ['kai_xin_shou_xia.png']  # 仅使用现有模板
            happy_coords = match_template_with_multiple_templates(image, happy_templates)
            if happy_coords:
                print("Detected '开心收下', clicking...")
                x, y = happy_coords
                subprocess.run([adb_path, 'shell', 'input', 'tap', str(x), str(y)])
                time.sleep(0.5)
                continue  # 点击后重新开始循环

            # 多个“继续”模板
            continue_templates = ['ji_xu.png']  # 仅使用现有模板
            continue_coords = match_template_with_multiple_templates(image, continue_templates)
            if continue_coords:
                print("Detected '继续', clicking...")
                x, y = continue_coords
                subprocess.run([adb_path, 'shell', 'input', 'tap', str(x), str(y)])
                time.sleep(0.8)
                continue  # 点击后重新开始循环

            # 多个“继续PK”模板
            continue_pk_templates = ['ji_xu_PK.png']  # 仅使用现有模板
            continue_pk_coords = match_template_with_multiple_templates(image, continue_pk_templates)
            if continue_pk_coords:
                print("Detected '继续PK', clicking...")
                x, y = continue_pk_coords
                subprocess.run([adb_path, 'shell', 'input', 'tap', str(x), str(y)])
                time.sleep(0.8)
                continue  # 点击后重新开始循环

            # 如果没有检测到任何按钮，等待一段时间再继续
            time.sleep(1)
        except RuntimeError as e:
            print(f"Error in button check thread: {e}")
            cv2.destroyAllWindows()


# 主流程
if __name__ == "__main__":
    # 检查 res/ 目录下的模板图像是否存在
    required_templates = ['kai_xin_shou_xia.png', 'ji_xu.png', 'ji_xu_PK.png']
    missing_templates = [tpl for tpl in required_templates if not os.path.isfile(os.path.join(template_dir, tpl))]
    if missing_templates:
        print("以下模板图像缺失，请确保 res/ 目录下存在这些文件：")
        for tpl in missing_templates:
            print(f" - {tpl}")
        exit(1)

    # 初始化调试图像
    debug_image = None

    # 运行检查按钮的多线程
    button_thread = threading.Thread(target=check_and_click_buttons)
    button_thread.daemon = True
    button_thread.start()

    try:
        while True:
            # 第一步：通过 adb 截取屏幕截图
            image = capture_screenshot()

            # 第二步：识别题目区域内的两个数字
            num1, num2 = recognize_two_numbers(image)

            # 进行比较并作答
            result = compare_numbers(num1, num2)

            # 输出比较结果并通过 ADB 划线
            answer_coords = (400, 520, 1200, 850)  # 作答区域坐标，需要根据实际情况调整
            # 在截屏上绘制作答区域
            cv2.rectangle(debug_image, (answer_coords[0], answer_coords[1]), (answer_coords[2], answer_coords[3]),
                          (255, 0, 255), 2)
            cv2.imshow('Debug Window', debug_image)
            cv2.waitKey(1)

            draw_result_on_screen(result, answer_coords)

            # 等待一段时间再进行下一次循环
            time.sleep(1)
    except KeyboardInterrupt:
        print("程序已停止。")
        cv2.destroyAllWindows()
    except RuntimeError as e:
        print(f"Error: {e}")
        cv2.destroyAllWindows()
