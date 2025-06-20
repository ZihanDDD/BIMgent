import platform
import os
import subprocess
import pyautogui
from conf.config import Config
from BIMgent.utils.dict_utils import kget
import time
config = Config()

env_path = kget(config.env_config, "env_path", default='')

class UIController():
    def __init__(self):
        pass

    def switch_env(self, env_name):
        current_os = platform.system()

        # @TODO: As the environment is tested on mac os. The windows environment is going to be developed

        current_os = "Windows"

        if current_os == "Windows":
            # Adjust the path if needed:
            vectorworks_path = fr"{env_path}"
            # Update the path based on your actual installation location.

            if os.path.exists(vectorworks_path):
                os.startfile(vectorworks_path)
            else:
                print("Path not found at:", vectorworks_path)

        elif current_os == "Darwin":  # macOS
            # On Mac, if WeChat is installed in the Applications folder:
            # This command attempts to open the "WeChat" application by name.
            subprocess.run(["open", "-a", env_name])
        else:
            print(f"{env_name} is not officially supported on this OS.")


class MouseController():

    def __init__(self):
        pass

    def move_mouse_to(self, x: int, y: int):
        """
        Moves the mouse pointer to the specified (x, y) location.
        """
        #print(f"Moving mouse to ({x}, {y})...")
        pyautogui.moveTo(x, y, duration=1)
        #print(f"Mouse moved to ({x}, {y}).")
        time.sleep(0.5)

    def left_click(self):
        """
        Performs a left-click at the current mouse location.
        """
        #print("Performing a left-click at the current location...")
        pyautogui.click(button='left')  
        #print("Left-click completed.")
        time.sleep(0.5)
        
        
    def type_name(self, name: str):
        time.sleep(0.5)  # Optional: short delay before typing
        pyautogui.typewrite(name, interval=0.05)  # interval adds natural typing delay

    
    def press_left_button(self):
        """
        Press and hold the left mouse button at the current location.
        """
        #print("Pressing and holding the left mouse button...")
        pyautogui.mouseDown(button='left')  # Press and hold the right mouse button
        #print("left mouse button is being held.")

    def release_left_button(self):
        """
        Releases the left mouse button.
        """
        #print("Releasing the left mouse button...")
        pyautogui.mouseUp(button='left')  # Release the right mouse button
        #print("left mouse button released.")
    
    def press_escape(self):
        """
        Simulates pressing the 'Esc' key.
        """
        #print("Pressing the 'Esc' key...")
        pyautogui.press('esc')  # Simulates pressing the 'Esc' key
        #print("'Esc' key pressed.")
        time.sleep(1)

    def delete(self):
        """
        Simulates pressing the 'Delete' key.
        """
        pyautogui.press('delete')
        time.sleep(1)
            
    
    def press_enter(self):
        """
        Simulates pressing the 'Esc' key.
        """

        #print("Pressing the 'enter' key...")
        pyautogui.press('enter')  # Simulates pressing the 'Esc' key
        #print("'enter' key pressed.")
        time.sleep(0.5)
        
    def shortcut(self, combo):
        if '+' in combo:
            # For combinations like "alt + shift + 2"
            keys = [k.strip() for k in combo.split('+')]
            
            # Pass the keys as separate arguments using unpacking
            pyautogui.hotkey(*keys)
        else:
            # For single keys like "9"
            key = combo.strip()
            pyautogui.press(key)
        time.sleep(1)
    
        
    def undo(self):
        print(f"Undo the previdous operation")
        pyautogui.hotkey('ctrl', 'z')


    def select_all(self):
        pyautogui.hotkey('ctrl', 'a')
        time.sleep(1)
        

