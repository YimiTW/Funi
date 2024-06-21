from selenium import webdriver
from selenium.webdriver.common.by import By
from googlesearch import search
import os
import time
import random

search_keyword = "who is harrison temple"
urls = search(search_keyword, num_results=5)

url_list = []

for url in urls:
    url_list.append(url)
    print(url)

options = webdriver.ChromeOptions()
options.add_argument('--incognito')
# options.add_argument('--headless')
options.add_argument("--enable-javascript")
options.add_argument('--remote-debugging-pipe')  
prefs = {
    "download.open_pdf_in_system_reader": False,
    "download.prompt_for_download": True,
    "plugins.always_open_pdf_externally": False,
    "download_restrictions": 3,
    "download.default_directory": 'NUL' if os.name == "nt" else '/dev/null',
}
options.add_experimental_option(
    "prefs", prefs
)
selenium_executable_path = os.path.join(os.getcwd(), "chromedriver" + (".exe" if os.name == "nt" else ""))
chrome_Service = webdriver.ChromeService(executable_path=selenium_executable_path)
driver = webdriver.Chrome(options=options, service=chrome_Service)

for i in range(len(url_list)):
    try:
        driver.get(url_list[i])
        print(driver.title)
    except:pass
    time.sleep(5)


