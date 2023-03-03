import time
from directkeys import PressKey, ReleaseKey, key_down, key_up
import time
import threading
import usb.core
import usb.util
import subprocess

subprocess.run(["runas", "/user:Administrator", "C:\\Users\\123\\Desktop\\test"])

direct_dic = {"UP": 45, "DOWN": 46, "LEFT": 44, "RIGHT": 47}

def move(direct, material=False, action_cache=None, press_delay=0.1, release_delay=0.1):
    print("向右移动")
    return "RIGHT"





