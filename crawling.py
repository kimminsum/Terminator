from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import urllib.request
import tkinter as tk
import time
import os
import sys
"""
Get Text
"""
window = tk.Tk()
window.title("ImageDownloader")
window.geometry("300x150+1000+100")
window.resizable(False, False)

# configure the grid
window.columnconfigure(0, weight=1)
window.columnconfigure(1, weight=3)

def get_text():
    global IMAGE_NAME
    global LIMIT_TRYS
    IMAGE_NAME = Image_name_entry.get()
    LIMIT_TRYS = Limit_trys_entry.get()
    window.destroy()

### Image_name
Image_name_label = tk.Label(window, text="Insert Image name:")
Image_name_label.grid(column=0, row=0, sticky=tk.W, padx=5, pady=5)

Image_name_entry = tk.Entry(window)
Image_name_entry.grid(column=1, row=0, sticky=tk.E, padx=5, pady=5)

### Limit_trys
Limit_trys_label = tk.Label(window, text="Insert Image number:")
Limit_trys_label.grid(column=0, row=1, sticky=tk.W, padx=5, pady=5)

Limit_trys_entry = tk.Entry(window)
Limit_trys_entry.grid(column=1, row=1, sticky=tk.E, padx=5, pady=5)

### get text button
login_button = tk.Button(window, text="Download", command=get_text)
login_button.grid(column=1, row=3, sticky=tk.E, padx=5, pady=5)

window.mainloop()

if IMAGE_NAME == "" or LIMIT_TRYS == "":
    sys.exit()
"""
Research Image
"""
driver = webdriver.Chrome("chromedriver.exe")
# options = webdriver.ChromeOptions()
# options.add_experimental_option('excludeSwitches', ['enable-logging'])
# driver = webdriver.Chrome(options=options)
driver.implicitly_wait(10)
driver.get("https://www.google.com/imghp?hl=ko&tab=8i")
### insert image name
RESEARCH_BOX = driver.find_element(By.CLASS_NAME, "gLFyf")
RESEARCH_BOX.send_keys(IMAGE_NAME) ##############################
RESEARCH_BOX.send_keys(Keys.RETURN)
"""
Get every Image
"""
SCROLL_PAUSE_SEC = 1
### get scroll height
last_height = driver.execute_script("return document.body.scrollHeight")

while True:
    ### scroll down to end
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    ### wait for 1 sec
    time.sleep(SCROLL_PAUSE_SEC)

    ### get scroll height again after scrolling down
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        try:
            driver.find_element(By.CSS_SELECTOR, "input.mye4qd").click()
        except:
            break
    last_height = new_height
"""
Download Image
"""
images = driver.find_elements(By.CSS_SELECTOR, "img.rg_i.Q4LuWd")
index = 0
file_name = ""
for image in images:
    if index == int(LIMIT_TRYS):
        break
    try:
        driver.implicitly_wait(2)
        image.click()
        image_link = str(driver.find_element(By.CSS_SELECTOR, "img.n3VNCb.KAlRDb").get_attribute("src"))
        ### distribute file name
        if ".jpg" in image_link:
            file_name = f"img/{index}.jpg"
        elif ".jpeg" in image_link:
            file_name = f"img/{index}.jpeg"
        elif ".png" in image_link:
            file_name = f"img/{index}.png"
        else:
            print("!!!Out of style!!!")
            continue
        ### download image in f"img/{number}.png"
        urllib.request.urlretrieve(str(image_link), str(file_name))
        index += 1
    except:
        print("!!!Image is blocked!!!")
        pass
os.system("cls")
driver.close()
