import time

import pyautogui


class MouseController():
    """Thin PyAutoGUI wrapper. Method names are the action vocabulary the
    action-generator LLM emits (see ``skill_executor.execute_actions``)."""

    def move_mouse_to(self, x: int, y: int):
        pyautogui.moveTo(x, y, duration=1)
        time.sleep(0.5)

    def left_click(self):
        pyautogui.click(button='left')
        time.sleep(0.5)

    def double_click(self):
        pyautogui.click(button='left')
        time.sleep(0.5)
        pyautogui.click(button='left')
        time.sleep(0.5)

    def type_name(self, name: str):
        time.sleep(0.5)
        pyautogui.typewrite(name, interval=0.05)

    def press_left_button(self):
        pyautogui.mouseDown(button='left')

    def release_left_button(self):
        pyautogui.mouseUp(button='left')

    def press_escape(self):
        pyautogui.press('esc')
        time.sleep(1)

    def delete(self):
        pyautogui.press('delete')
        time.sleep(1)

    def press_enter(self):
        pyautogui.press('enter')
        time.sleep(0.5)

    def shortcut(self, combo):
        """``"alt + shift + 2"`` -> hotkey; ``"9"`` -> single key press."""
        if '+' in combo:
            keys = [k.strip() for k in combo.split('+')]
            pyautogui.hotkey(*keys)
        else:
            pyautogui.press(combo.strip())
        time.sleep(3)

    def undo(self):
        print("Undo the previous operation")
        pyautogui.hotkey('ctrl', 'z')

    def select_all(self):
        pyautogui.hotkey('ctrl', 'a')
        time.sleep(1)
