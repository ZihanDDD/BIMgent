from pynput import mouse, keyboard

# 控制鼠标监听器的变量
mouse_listener = None

def on_move(x, y):
    print(f"Mouse moved to ({x}, {y})")

def on_click(x, y, button, pressed):
    print(f"Mouse {'pressed' if pressed else 'released'} at ({x}, {y}) with {button}")

def on_scroll(x, y, dx, dy):
    print(f"Mouse scrolled at ({x}, {y}) with delta ({dx}, {dy})")

# 当按下键盘时触发
def on_press(key):
    try:
        if key.char == 'x':
            print("Pressed 'x'. Exiting...")
            mouse_listener.stop()
            return False  
    except AttributeError:
        pass  


mouse_listener = mouse.Listener(
    on_move=on_move,
    on_click=on_click,
    on_scroll=on_scroll)

keyboard_listener = keyboard.Listener(
    on_press=on_press)

mouse_listener.start()
keyboard_listener.start()

mouse_listener.join()
keyboard_listener.join()