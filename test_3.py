import time
import threading
import usb.core
import usb.util

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
    channels = {'up': 45, 'right': 47, 'left': 44, 'down': 46}
    while True:
        direction = input("请输入方向（up/right/left/down）：")
        channel = channels.get(direction)
        if channel is None:
            print("输入有误，请重新输入")
        else:
            send_control_signal(channel, 1)
            time.sleep(0.5)
            send_control_signal(channel, 0)
            time.sleep(0.5)

capture_thread = threading.Thread(target=capture_and_send, args=(dev,))
capture_thread.start()


#
###我想引用同目录里的direction_move 中的move函数,让channels=move.  def move(direct, material=False, action_cache=None, press_delay=0.1, release_delay=0.1):。让channels=move