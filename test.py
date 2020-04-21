import win32api

# import ctypes

# awareness = ctypes.c_int()
# ctypes.windll.shcore.SetProcessDpiAwareness(0)

prevPos = (-1,-1)
while True:
    curPos = win32api.GetCursorPos()
    if prevPos != curPos:
        prevPos = curPos
        print(curPos)
