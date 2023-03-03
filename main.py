import numpy as np
import sys
sys.path.append('/home/user/myproject')
from grabscreen import grab_screen
import cv2
import time
import directkeys
import torch
from torch.autograd import Variable
from directkeys import PressKey, ReleaseKey, key_down, key_up
from getkeys import key_check
from utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, xywh2xyxy, plot_one_box, strip_optimizer, set_logging)
from models.experimental import attempt_load
from direction_move import move
from small_recgonize import current_door, next_door
from skill_recgnize import skill_rec
import random
import subprocess
import time
import threading
import usb.core
import usb.util
from direction_move import move



#

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
    #   img：待处理的图像       new_shape：缩放后的矩形大小，默认为 (640, 640)          color：添加的边框颜色，默认为 (114, 114, 114)
    #   auto：是否按照 64 的倍数自动调整矩形大小，默认为 False          scaleFill：是否拉伸图像以填充整个矩形，默认为 False
    #   scaleup：是否允许将图像缩放到大于原始大小，默认为 True
    #   定义名为 letterbox 的函数，用于将图像缩放成指定大小的矩形
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]      #   获取输入图像的高度和宽度
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)  # 如果new_shape是一个整数，则将其转化为包含两个相同值的元组。
    #       定义名为 letterbox 的函数，用于将图像缩放成指定大小的矩形

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)
    #       计算缩放比例r，其中r是新的图像大小与原图像大小的比值。如果scaleup为False，则将r限制在1.0以下，以避免放大图像。
    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    #   计算目标图像大小和填充后的图像大小之间的比例，以及需要添加的填充边框大小。

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios
    #       如果auto为True，则自动调整图像大小，使得宽度和高度均为32的倍数。如果scaleFill为True，则将图像拉伸以填充整个目标矩形。

    dw /= 2  # divide padding into 2 sides
    dh /= 2
    #   将填充大小分配到两侧，得到左侧和右侧、上侧和下侧的填充大小。
    if shape[::-1] != new_unpad:  # resize  #   如果当前图像的大小与调整大小后的大小不同，则执行下面的代码。
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)  # 调整图像的大小为调整后的大小，采用双线性插值方法。
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))  # 计算需要添加的上下边框大小，使用四舍五入的方式将小数点后的数值转化为整数
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))  # 计算需要添加的左右边框大小，同样采用四舍五入的方式将小数点后的数值转化为整数
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    #   为图小像添加边框，上边框大小为 top，下边框大为 bottom
    #   左边框大小为 left，右边框大小为 right，边框的类型为常数边框，颜色为 color

    return img, ratio, (dw, dh)  # 返回调整大小和添加边框后的图像，调整比例 ratio，边框大小 (dw, dh)


# 设置所有用到的参数
weights = r'C:\Users\123\Desktop\yolov5jihe\exp6\weights\best.pt'  # yolo5 模型存放的位置
# weights = r'F:\Computer_vision\yolov5\YOLO5\yolov5-master\runs\exp0\weights\best.pt'
#device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
#device = torch.device("cpu")
#model = attempt_load(weights, map_location=device)  # load FP32 model
#model = attempt_load(weights, device=device, inplace=True, fuse=True)
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
model = torch.load(weights, map_location=device)["model"].float().fuse().eval()

window_size = (0, 0, 1280, 800)  # 截屏的位置
img_size = 800  # 输入到yolo5中的模型尺寸
paused = False

# if device.type != 'cpu':
#     half = True
# else:
#     half = False

# half = device.type != 'cpu'  # device.type != 'cpu'  True
half = True
view_img = True  # 是否观看目标检测结果
save_txt = False
conf_thres = 0.3  # NMS的置信度过滤
iou_thres = 0.2  # NMS的IOU阈值
classes = None
agnostic_nms = False  # 不同类别的NMS时也参数过滤
skill_char = "XYHGXFAXDSWXETX"  # 技能按键，使用均匀分布随机抽取
direct_dic = {"UP": 0xC8, "DOWN": 0xD0, "LEFT": 0xCB, "RIGHT": 0xCD}  # 上下左右的键码
names = ['hero', 'small_map','monster','money','material','door','BOSS','box,','options']  # 所有类别名
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
if half:
    model.half()  # to FP16
action_cache = None  # 动作标记
press_delay = 0.1  # 按压时间
release_delay = 0.1  # 释放时间
# last_time = time.time()
frame = 0  # 帧
door1_time_start = -20
next_door_time = -20
fs = 1  # 每四帧处理一次

# 倒计时
for i in list(range(5))[::-1]:
    print(i + 1)
    time.sleep(1)

# 捕捉画面+目标检测+玩游戏
while True:
    if not paused:  # 屏幕实时捕捉          如果没暂停
        t_start = time.time()  # 获取当前时间
        img0 = grab_screen(window_size)  # 截取屏幕图像
        frame += 1  # 帧数加1
        if frame % fs == 0:  # 如果帧数可以被处理步长整除
            # img0 = cv2.imread("datasets/guiqi/yolo5_datasets/imgs/1004_14.jpg")
            img0 = cv2.cvtColor(img0, cv2.COLOR_BGRA2BGR)  # 如果帧数可以被处理步长整除
            # Padded resize
            img = letterbox(img0, new_shape=img_size)[0]  # 对图像进行填充缩放，获取新的图像和缩放比例

            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB      # 将 BGR 通道顺序转换成 RGB 顺序，并调换轴的顺序
            img = np.ascontiguousarray(img)

            img = torch.from_numpy(img).to(device).unsqueeze(0)  # 将 numpy 数组转换成 PyTorch 张量，再增加一维
            img = img.half() if half else img.float()  # uint8 to fp16/32       #        将数据类型转换为 half 或者 float
            img /= 255.0  # 0 - 255 to 0.0 - 1.0        #       # 将像素值从 [0, 255] 转换到 [0, 1]

            pred = model(img, augment=False)[0]  # # 进行预测
            #   上面的代码是一个处理视频或实时摄像头的循环中的一部分。首先，如果程序没有暂停，就会获取当前时间并截取屏幕图像
            #   然后，帧数加 1，并检查是否需要处理该帧。如果需要处理，就将图像颜色通道从 BGRA 转换为 BGR，并进行填充缩放，获取新的图像和缩放比例
            #   接下来，将图像颜色通道顺序从 BGR 转换为 RGB，并调换轴的顺序，然后将 numpy 数组转换为 PyTorch 张量，并增加一个维度
            #   然后，将数据类型转换为 half 或者 float，并将像素值从 [0, 255] 转换到 [0, 1]。最后，使用模型对图像进行预测。

            # Apply NMS
            det = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
            #   使用non_max_suppression函数对模型预测结果进行非极大值抑制，得到剩余的预测框列表
            gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
            #   计算比例因子，用于将缩放后的预测框坐标还原到原始图像上
            det = det[0]  # 取出预测框列表中的第一个元素，这里默认为只有一个元素
            if det is not None and len(det):  # 如果预测框列表不为空，则继续执行下面的操作
                # Rescale boxes from img_size to im0 size
                ###det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round() #原代码
                det[:, :4] = torch.tensor(np.array(scale_coords(det[:, :4], img.shape[2:], img0.shape)).round(), dtype=torch.float32, device=device)
                #det[:, :4] = torch.tensor(np.array(scale_coords(det[:, :4], img.shape[2:], img0.shape)).round(),
                #                          dtype=torch.float32, device=device).to(device) #用GPU

                #   使用scale_coords函数将缩放后的预测框坐标还原到原始图像上，并四舍五入取整。

                # Print results
                for c in det[:, -1].unique():  # 计算每个类别的检测数量n
                    n = (det[:, -1] == c).sum()  # detections per class

                img_object = []  # 初始化图像对象和类别对象列表
                cls_object = []
                # Write results
                hero_conf = 0  # 初始化英雄的置信度和索引
                hero_index = 0
                for idx, (*xyxy, conf, cls) in enumerate(reversed(det)):
                    #   遍历预测框列表中的每个元素，其中reversed函数用于将预测框按照置信度从大到小排序。

                    # if save_txt:  # Write to file
                    #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    #     with open(txt_path + '.txt', 'a') as f:
                    #         f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
                    cls = int(cls)
                    img_object.append(xywh)
                    cls_object.append(names[cls])


                    #  将预测框坐标转换为中心点坐标和宽高，将类别编号转换为类别名称，并将结果保存到图像对象和类别对象列表中

                    if names[cls] == "hero" and conf > hero_conf:
                        hero_conf = conf
                        hero_index = idx
                    #  如果检测到的物体是"hero"且置信度高于之前检测到的"hero"的置信度，将其视为当前帧的主角，并更新"hero_conf"和"hero_index"

                    if view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=2)
                    #   如果需要在图像上显示检测框，将框画在图像上，框的标签为物体类别和置信度，框的颜色根据类别而定，框的线宽为2

                # 游戏
                thx = 30  # 捡东西时，x方向的阈值
                thy = 30  # 捡东西时，y方向的阈值
                attx = 150  # 攻击时，x方向的阈值
                atty = 50  # 攻击时，y方向的阈值

                #   这段代码是一个游戏机器人脚本，主要作用是实现自动化游戏操作
                if current_door(img0) == 1 and time.time() - door1_time_start > 10:
                    door1_time_start = time.time()
                    # move(direct="RIGHT", action_cache=action_cache, press_delay=press_delay,
                    #      release_delay=release_delay)
                    # ReleaseKey(direct_dic["RIGHT"])
                    # directkeys.key_press("SPACE")
                    directkeys.key_press("CTRL")
                    time.sleep(1)
                    directkeys.key_press("ALT")
                    time.sleep(0.5)
                    action_cache = None
                    #   如果当前屏幕上有打开的门并且距离上次开门时间已经超过10秒，那么机器人会在屏幕上按下“CTRL”和“ALT”键，然后将之前存储的动作缓存清空

                # 扫描英雄
                if "hero" in cls_object:
                    # hero_xywh = img_object[cls_object.index("hero")]
                    hero_xywh = img_object[hero_index]
                    cv2.circle(img0, (int(hero_xywh[0]), int(hero_xywh[1])), 1, (0, 0, 255), 10)
                    #   如果屏幕上存在“hero”（英雄）这个物体，则将其坐标保存到变量hero_xywh中，并在屏幕上以红色画一个小圆6点表示英雄的位置

                    # print(hero_index)
                    # print(cls_object.index("hero"))
                else:
                    continue
                # 打怪
                if "monster" in cls_object or "BOSS" in cls_object:  # 判断是否检测到怪物或BOSS
                    min_distance = float("inf")
                    for idx, (c, box) in enumerate(zip(cls_object, img_object)):
                        if c == 'monster' or c == "BOSS":  # 然后，将英雄的位置坐标和检测到的怪物/BOSS的位置坐标计算距离，以找到距离英雄最近的怪物/BOSS。
                            dis = ((hero_xywh[0] - box[0]) ** 2 + (hero_xywh[1] - box[1]) ** 2) ** 0.5
                            if dis < min_distance:  # 距离比较
                                monster_box = box  # 如果当前怪物/BOSS比之前检测到的所有怪物/BOSS都要近，
                                monster_index = idx  # 则更新min_distance、monster_box和monster_index变量
                                min_distance = dis
                    if abs(hero_xywh[0] - monster_box[0]) < attx and abs(hero_xywh[1] - monster_box[1]) < atty:
                        #   如果怪物/BOSS距离英雄不足attx和atty，则开始攻击怪物/BOSS
                        if "BOSS" in cls_object:  # 如果检测到的对象是BOSS
                            directkeys.key_press("R")  # 则使用directkeys.key_press()函数按键攻击技能（'R'和'Q'键
                            directkeys.key_press("Q")
                            # time.sleep(0.5)
                            skill_name = skill_char[int(np.random.randint(len(skill_char), size=1)[0])]
                            #   随机选择一个技能名，直到检测到该技能的特征。如果检测到技能特征，则按键攻击技能3次，然后退出while循环。
                            #   否则，继续选择另一个随机技能，直到找到一个可以使用的技能

                            while True:  # while True: 表示这是一个无限循环，只要条件不满足，就会一直执行下去
                                if skill_rec(skill_name, img0):
                                    directkeys.key_press(skill_name)  # 使用三次技能保证一定能触发
                                    directkeys.key_press(skill_name)
                                    directkeys.key_press(skill_name)
                                    break
                                else:  # 表示如果检测不到需要攻击的怪物，就重新选择技能并等待机会
                                    skill_name = skill_char[int(np.random.randint(len(skill_char), size=1)[0])]
                                    # 表示重新选择技能。skill_char是一个列表，包含了所有可以使用的技能名称，np.random.randint(len(skill_char), size=1)[0]
                                    # 表示在这个列表中随机选择一个技能的索引。然后将这个技能名称赋值给 skill_name 变量，等待下一次检测

                        else:
                            skill_name = skill_char[int(np.random.randint(len(skill_char), size=1)[0])]
                            while True:
                                if skill_rec(skill_name, img0):
                                    directkeys.key_press(skill_name)
                                    directkeys.key_press(skill_name)
                                    directkeys.key_press(skill_name)
                                    break
                                else:
                                    skill_name = skill_char[int(np.random.randint(len(skill_char), size=1)[0])]
                        print("释放技能攻击")
                        if not action_cache:
                            pass
                        elif action_cache not in ["LEFT", "RIGHT", "UP", "DOWN"]:
                            ReleaseKey(direct_dic[action_cache.strip().split("_")[0]])
                            ReleaseKey(direct_dic[action_cache.strip().split("_")[1]])
                            action_cache = None
                        elif action_cache:
                            ReleaseKey(direct_dic[action_cache])
                            action_cache = None
                        # break
                    elif monster_box[1] - hero_xywh[1] < 0 and monster_box[0] - hero_xywh[0] > 0:
                        if abs(monster_box[1] - hero_xywh[1]) < thy:
                            action_cache = move(direct="RIGHT", material=True, action_cache=action_cache,
                                                press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                        elif hero_xywh[1] - monster_box[1] < monster_box[0] - hero_xywh[0]:
                            action_cache = move(direct="RIGHT_UP", material=True, action_cache=action_cache,
                                                press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                        elif hero_xywh[1] - monster_box[1] >= monster_box[0] - hero_xywh[0]:
                            action_cache = move(direct="UP", material=True, action_cache=action_cache,
                                                press_delay=press_delay,
                                                release_delay=release_delay)
                        # break
                    elif monster_box[1] - hero_xywh[1] < 0 and monster_box[0] - hero_xywh[0] < 0:
                        if abs(monster_box[1] - hero_xywh[1]) < thy:
                            action_cache = move(direct="LEFT", material=True, action_cache=action_cache,
                                                press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                        elif hero_xywh[1] - monster_box[1] < hero_xywh[0] - monster_box[0]:
                            action_cache = move(direct="LEFT_UP", material=True, action_cache=action_cache,
                                                press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                        elif hero_xywh[1] - monster_box[1] >= hero_xywh[0] - monster_box[0]:
                            action_cache = move(direct="UP", material=True, action_cache=action_cache,
                                                press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                    elif monster_box[1] - hero_xywh[1] > 0 and monster_box[0] - hero_xywh[0] < 0:
                        if abs(monster_box[1] - hero_xywh[1]) < thy:
                            action_cache = move(direct="LEFT", material=True, action_cache=action_cache,
                                                press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                        elif monster_box[1] - hero_xywh[1] < hero_xywh[0] - monster_box[0]:
                            action_cache = move(direct="LEFT_DOWN", material=True, action_cache=action_cache,
                                                press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                        elif monster_box[1] - hero_xywh[1] >= hero_xywh[0] - monster_box[0]:
                            action_cache = move(direct="DOWN", material=True, action_cache=action_cache,
                                                press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                    elif monster_box[1] - hero_xywh[1] > 0 and monster_box[0] - hero_xywh[0] > 0:
                        if abs(monster_box[1] - hero_xywh[1]) < thy:
                            action_cache = move(direct="RIGHT", material=True, action_cache=action_cache,
                                                press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                        elif monster_box[1] - hero_xywh[1] < monster_box[0] - hero_xywh[0]:
                            action_cache = move(direct="RIGHT_DOWN", material=True, action_cache=action_cache,
                                                press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                        elif monster_box[1] - hero_xywh[1] >= monster_box[0] - hero_xywh[0]:
                            action_cache = move(direct="DOWN", material=True, action_cache=action_cache,
                                                press_delay=press_delay,
                                                release_delay=release_delay)
                            # break

                # 移动到下一个地图
                if "door" in cls_object and "monster" not in cls_object and "BOSS" not in cls_object and "material" not in cls_object and "money" not in cls_object:
                    for idx, (c, box) in enumerate(zip(cls_object, img_object)):
                        if c == 'door':
                            door_box = box
                            door_index = idx
                    if door_box[0] < img0.shape[0] // 2:
                        action_cache = move(direct="RIGHT", action_cache=action_cache, press_delay=press_delay,
                                            release_delay=release_delay)
                        # break
                    elif door_box[1] - hero_xywh[1] < 0 and door_box[0] - hero_xywh[0] > 0:
                        if abs(door_box[1] - hero_xywh[1]) < thy and abs(door_box[0] - hero_xywh[0]) < thx:
                            action_cache = None
                            print("进入下一地图")
                            # break
                        elif abs(door_box[1] - hero_xywh[1]) < thy:
                            action_cache = move(direct="RIGHT", action_cache=action_cache, press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                        elif hero_xywh[1] - door_box[1] < door_box[0] - hero_xywh[0]:
                            action_cache = move(direct="RIGHT_UP", action_cache=action_cache, press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                        elif hero_xywh[1] - door_box[1] >= door_box[0] - hero_xywh[0]:
                            action_cache = move(direct="UP", action_cache=action_cache, press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                    elif door_box[1] - hero_xywh[1] < 0 and door_box[0] - hero_xywh[0] < 0:
                        if abs(door_box[1] - hero_xywh[1]) < thy and abs(door_box[0] - hero_xywh[0]) < thx:
                            action_cache = None
                            print("进入下一地图")
                            # break
                        elif abs(door_box[1] - hero_xywh[1]) < thy:
                            action_cache = move(direct="LEFT", action_cache=action_cache, press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                        elif hero_xywh[1] - door_box[1] < hero_xywh[0] - door_box[0]:
                            action_cache = move(direct="LEFT_UP", action_cache=action_cache, press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                        elif hero_xywh[1] - door_box[1] >= hero_xywh[0] - door_box[0]:
                            action_cache = move(direct="UP", action_cache=action_cache, press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                    elif door_box[1] - hero_xywh[1] > 0 and door_box[0] - hero_xywh[0] < 0:
                        if abs(door_box[1] - hero_xywh[1]) < thy and abs(door_box[0] - hero_xywh[0]) < thx:
                            action_cache = None
                            print("进入下一地图")
                            # break
                        elif abs(door_box[1] - hero_xywh[1]) < thy:
                            action_cache = move(direct="LEFT", action_cache=action_cache, press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                        elif door_box[1] - hero_xywh[1] < hero_xywh[0] - door_box[0]:
                            action_cache = move(direct="LEFT_DOWN", action_cache=action_cache, press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                        elif door_box[1] - hero_xywh[1] >= hero_xywh[0] - door_box[0]:
                            action_cache = move(direct="DOWN", action_cache=action_cache, press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                    elif door_box[1] - hero_xywh[1] > 0 and door_box[0] - hero_xywh[0] > 0:
                        if abs(door_box[1] - hero_xywh[1]) < thy and abs(door_box[0] - hero_xywh[0]) < thx:
                            action_cache = None
                            print("进入下一地图")
                            # break
                        elif abs(door_box[1] - hero_xywh[1]) < thy:
                            action_cache = move(direct="RIGHT", action_cache=action_cache, press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                        elif door_box[1] - hero_xywh[1] < door_box[0] - hero_xywh[0]:
                            action_cache = move(direct="RIGHT_DOWN", action_cache=action_cache, press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                        elif door_box[1] - hero_xywh[1] >= door_box[0] - hero_xywh[0]:
                            action_cache = move(direct="DOWN", action_cache=action_cache, press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                if "money" not in cls_object and "material" not in cls_object and "monster" not in cls_object \
                        and "BOSS" not in cls_object and "door" not in cls_object and 'box' not in cls_object \
                        and 'options' not in cls_object:
                    # if next_door(img0) == 0 and abs(time.time()) - next_door_time > 10:
                    #     next_door_time = time.time()
                    #     action_cache = move(direct="LEFT", action_cache=action_cache, press_delay=press_delay,
                    #                         release_delay=release_delay)
                    #     # time.sleep(3)
                    # else:
                    #     action_cache = move(direct="RIGHT", action_cache=action_cache, press_delay=press_delay,
                    #                     release_delay=release_delay)

                    action_cache = move(direct="RIGHT", action_cache=action_cache, press_delay=press_delay,
                                        release_delay=release_delay)
                    # break

                # 捡材料
                if "monster" not in cls_object and "hero" in cls_object and (
                        "material" in cls_object or "money" in cls_object):
                    min_distance = float("inf")
                    hero_xywh[1] = hero_xywh[1] + (hero_xywh[3] // 2) * 0.7
                    thx = thx / 2
                    thy = thy / 2
                    for idx, (c, box) in enumerate(zip(cls_object, img_object)):
                        if c == 'material' or c == "money":
                            dis = ((hero_xywh[0] - box[0]) ** 2 + (hero_xywh[1] - box[1]) ** 2) ** 0.5
                            if dis < min_distance:
                                material_box = box
                                material_index = idx
                                min_distance = dis
                    if abs(material_box[1] - hero_xywh[1]) < thy and abs(material_box[0] - hero_xywh[0]) < thx:
                        if not action_cache:
                            pass
                        elif action_cache not in ["LEFT", "RIGHT", "UP", "DOWN"]:
                            ReleaseKey(direct_dic[action_cache.strip().split("_")[0]])
                            ReleaseKey(direct_dic[action_cache.strip().split("_")[1]])
                            action_cache = None
                        else:
                            ReleaseKey(direct_dic[action_cache])
                            action_cache = None
                        time.sleep(1)
                        directkeys.key_press("X")
                        print("捡东西")
                        # break

                    elif material_box[1] - hero_xywh[1] < 0 and material_box[0] - hero_xywh[0] > 0:

                        if abs(material_box[1] - hero_xywh[1]) < thy:
                            action_cache = move(direct="RIGHT", material=True, action_cache=action_cache,
                                                press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                        elif hero_xywh[1] - material_box[1] < material_box[0] - hero_xywh[0]:
                            action_cache = move(direct="RIGHT_UP", material=True, action_cache=action_cache,
                                                press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                        elif hero_xywh[1] - material_box[1] >= material_box[0] - hero_xywh[0]:
                            action_cache = move(direct="UP", material=True, action_cache=action_cache,
                                                press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                    elif material_box[1] - hero_xywh[1] < 0 and material_box[0] - hero_xywh[0] < 0:
                        if abs(material_box[1] - hero_xywh[1]) < thy:
                            action_cache = move(direct="LEFT", material=True, action_cache=action_cache,
                                                press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                        elif hero_xywh[1] - material_box[1] < hero_xywh[0] - material_box[0]:
                            action_cache = move(direct="LEFT_UP", material=True, action_cache=action_cache,
                                                press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                        elif hero_xywh[1] - material_box[1] >= hero_xywh[0] - material_box[0]:
                            action_cache = move(direct="UP", material=True, action_cache=action_cache,
                                                press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                    elif material_box[1] - hero_xywh[1] > 0 and material_box[0] - hero_xywh[0] < 0:
                        if abs(material_box[1] - hero_xywh[1]) < thy:
                            action_cache = move(direct="LEFT", material=True, action_cache=action_cache,
                                                press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                        elif material_box[1] - hero_xywh[1] < hero_xywh[0] - material_box[0]:
                            action_cache = move(direct="LEFT_DOWN", material=True, action_cache=action_cache,
                                                press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                        elif material_box[1] - hero_xywh[1] >= hero_xywh[0] - material_box[0]:
                            action_cache = move(direct="DOWN", material=True, action_cache=action_cache,
                                                press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                    elif material_box[1] - hero_xywh[1] > 0 and material_box[0] - hero_xywh[0] > 0:
                        if abs(material_box[1] - hero_xywh[1]) < thy:
                            action_cache = move(direct="RIGHT", material=True, action_cache=action_cache,
                                                press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                        elif material_box[1] - hero_xywh[1] < material_box[0] - hero_xywh[0]:
                            action_cache = move(direct="RIGHT_DOWN", material=True, action_cache=action_cache,
                                                press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                        elif material_box[1] - hero_xywh[1] >= material_box[0] - hero_xywh[0]:
                            action_cache = move(direct="DOWN", material=True, action_cache=action_cache,
                                                press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                # 开箱子
                if "box" in cls_object:
                    box_num = 0
                    for b in cls_object:
                        if b == "box":
                            box_num += 1
                    if box_num >= 4:
                        directkeys.key_press("ESC")
                        print("打开箱子ESC")
                        # break62

                # 重新开始
                time_option = -20
                if "options" in cls_object:
                    if not action_cache:
                        pass
                    elif action_cache not in ["LEFT", "RIGHT", "UP", "DOWN"]:
                        ReleaseKey(direct_dic[action_cache.strip().split("_")[0]])
                        ReleaseKey(direct_dic[action_cache.strip().split("_")[1]])
                        action_cache = None
                    else:
                        ReleaseKey(direct_dic[action_cache])
                        action_cache = None
                    if time.time() - time_option > 10:
                        directkeys.key_press("NUM0")
                        print("移动物品到脚下")
                        directkeys.key_press("X")
                        time_option = time.time()
                    directkeys.key_press("F2")
                    print("重新开始F2")
                    # break
            t_end = time.time()
            #print("一帧游戏操作所用时间：", (t_end - t_start) / fs)

            img0 = cv2.resize(img0, (600, 375))         # 实时画面的尺寸
            # Stream results
            if view_img:
                cv2.imshow('window', img0)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    raise StopIteration

    # Setting pause and unpause
    keys = key_check()
    if 'P' in keys:
        if not action_cache:
            pass
        elif action_cache not in ["LEFT", "RIGHT", "UP", "DOWN"]:
            ReleaseKey(direct_dic[action_cache.strip().split("_")[0]])
            ReleaseKey(direct_dic[action_cache.strip().split("_")[1]])
            action_cache = None
        else:
            ReleaseKey(direct_dic[action_cache])
            action_cache = None
        if paused:
            paused = False
            time.sleep(1)
        else:
            paused = True
            time.sleep(1)


dev = usb.core.find(idVendor=0x2704, idProduct=0x2017)
if dev is None:
    raise ValueError("Device not found")
else:
    print("Device found")

dev.set_configuration()

def send_control_signal(channel, state):
    data = [0x03, channel, state]
    dev.write(1, data)
    print(f"Channel {channel} is {'on' if state == 1 else 'off'}.")

def capture_and_send(dev):
    action_cache = None  # 初始化动作缓存
    press_delay = release_delay = 0.1  # 按下和释放的延迟都为0.1秒
    while True:
        direction = input("请输入方向（up/right/left/down）：")
        channels = move(direction, action_cache=action_cache, press_delay=press_delay, release_delay=release_delay)
        action_cache = channels[-1] if channels else None  # 更新动作缓存
        if not channels:
            print("输入有误，请重新输入")
        else:
            for channel in channels:
                send_control_signal(channel, 1)
                time.sleep(0.5)
                send_control_signal(channel, 0)

capture_thread = threading.Thread(target=capture_and_send, args=(dev,))
capture_thread.start()
