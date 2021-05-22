from selenium import webdriver
from bs4 import BeautifulSoup
import requests as re

# path_to_driver = r'U:\\git\\hff-prediction\\chromedriver.exe'
path_to_driver_ie = r'U:\\git\\hff-prediction\\IEDriverServer.exe'
#browser = webdriver.Chrome(executable_path=path_to_driver)
browser = webdriver.Ie(executable_path=path_to_driver_ie)
driver_path = webdriver.Ie

weather_predictions_url = 'http://www.buienradar.nl/weer/debilt/nl/2757783/14daagse'

test = browser.get(weather_predictions_url)

#browser.find_element_by_link_text("Akkoord").click()
#browser.find_element_by_link_text("Nu niet, misschien later")

