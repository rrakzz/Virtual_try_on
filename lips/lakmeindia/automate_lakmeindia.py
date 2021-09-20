import warnings
from datetime import datetime
from time import sleep
import pyautogui
import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from selenium import webdriver
from bs4 import BeautifulSoup
warnings.filterwarnings('ignore')
from selenium.webdriver.common.keys import Keys
from sys import argv
import requests  # to get image from the web
import shutil  # to save it locally
starttime = datetime.now()

def downloadshadelink_image(image_url, shadecolor, output_images_folder__path):
    try:
        if '?' in image_url:
            image_url = image_url[0: image_url.index('?')]
            image_url =  image_url.strip()
        # filename = image_url.split("/")[-1]
        filename = output_images_folder__path + shadecolor + '_thumb.jpg'
        print(output_images_folder__path + shadecolor + 'thumb.jpg')
        # Open the url image, set stream to True, this will return the stream content.
        r = requests.get(image_url, stream=True)
        # Check if the image was retrieved successfully
        if r.status_code == 200:
            # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
            r.raw.decode_content = True
            # Open a local file with wb ( write binary ) permission.
            with open(filename, 'wb') as f:
                shutil.copyfileobj(r.raw, f)

        return filename
    except Exception as e:
        print(e)
def removeSpecialCharacter(s):
    t = ""
    for i in s:
        if (i >= 'A' and i <= 'Z') or (i >= 'a' and i <= 'z'):
            t += i
    return t

configfile = 'input.xlsx'
outputfile = 'outputfile.csv'
input_maincategory = 'lips'
configdata = pd.read_excel(configfile, sheet_name='Config')
config_dict = {}
for index, row in configdata.iterrows():
    # print(row['Variables Name'])
    try:
        var_name = str(row['Variables Name'])
        config_dict[var_name] = str(row['Website1'])
    except:
        continue
# print(config_dict)


options = webdriver.ChromeOptions()
options.add_experimental_option('useAutomationExtension', False)
outputfile_path = config_dict['output_path']
# if outputfile_path.lower() in config_dict:
prefs = {'download.default_directory' : outputfile_path}
options.add_experimental_option('prefs', prefs)
driver = webdriver.Chrome(options=options)
driver.maximize_window()
website_str = 'web_name'
if website_str.lower() in config_dict:
    websitename = config_dict[website_str.lower()]
    driver.get(websitename)
    sleep(20)
try:
    popyp_close = 'pop_close'
    if popyp_close.lower() in config_dict:
        popyp_close_str = config_dict[popyp_close.lower()]
        driver.find_element_by_css_selector(popyp_close_str).click()
except:
    pass
sleep(5)
try:
    cookies_close = 'cookies_close'
    if cookies_close.lower() in config_dict:
        cookies_element_path = config_dict[cookies_close.lower()]
        cookies_element = driver.find_element_by_xpath(cookies_element_path)
        cookies_element.click()
except:
    pass
driver.switch_to.frame(driver.find_element_by_tag_name('iframe'))

try:
    imageupload = 'upload_image'
    if imageupload.lower() in config_dict:
        imageupload_path = config_dict[imageupload.lower()]
        driver.find_element_by_css_selector(imageupload_path).click()
except:
    pass
sleep(6)

testimage = 'test_path'
if testimage.lower() in config_dict:
    testimage_path = config_dict[testimage.lower()]
    pyautogui.write(testimage_path)
    sleep(5)
    pyautogui.press('enter')
    sleep(5)
iframe_path = config_dict['iframe_path']
framecontent = driver.find_element_by_xpath(iframe_path)
# print(framecontent)
iframe_container = config_dict['iframe_container']
data_radiums = driver.find_elements_by_xpath(iframe_container)
# print(len(data_radiums))
data_radium_flag = False
for data_radium in data_radiums:
    try:
        iframe_images_path = config_dict['iframe_images']
        imagetags = data_radium.find_elements_by_xpath(iframe_images_path)
        # print(len(imagetags))
        for k_tag in range(0, len(imagetags)):
            if k_tag == 4:
                origialimage_download = imagetags[k_tag]
                origialimage_download.click()
                sleep(10)
                data_radium_flag = True
                break
        if data_radium_flag == True:
            break
    except:
        continue
output_images_folder = 'out_folder'
output_images_folder__path = config_dict[output_images_folder.lower()]
resultpath = output_images_folder__path
originalimage_folder = outputfile_path
originalimagepath = 'originalimage.jpg'
try:
    os.rename(originalimage_folder + 'ymk-beauty.jpg',originalimage_folder + originalimagepath)
except:
    pass
try:
    shutil.copy(originalimage_folder + originalimagepath, resultpath)
except:
    pass
try:
    os.remove(originalimage_folder + originalimagepath)
except:
    pass
filelist = [f for f in os.listdir(originalimage_folder) if f.endswith(".jpg")]
for f in filelist:
    os.remove(os.path.join(originalimage_folder, f))
driver.switch_to.default_content()
# driver.find_element_by_xpath("//button[contains(@id,'onetrust-accept-btn-handler')]")
category_str = 'cat1_xpath'
category_path = config_dict[category_str.lower()]
# tablinks = driver.find_elements_by_xpath('//button[@class="tablinks"]')
tablinks = driver.find_elements_by_xpath(category_path)
virtual_img_str = 'virtual_img'
virtual_img_path = config_dict[virtual_img_str.lower()]
virtualimg = driver.find_element_by_xpath(virtual_img_path)
driver.execute_script("arguments[0].scrollIntoView();", virtualimg)
# driver.execute_script("window.scrollTo(0, 350);")
tab_count = 0
resultdata_df = []
temp = pd.DataFrame({'Category': 'Category element', 'Sub Category':'Sub Category element', 'Product Name': 'Product Name', 'Shade Name': 'Shade Name',}, index=[0])
for tablink in tablinks:
    tab_str = str(tablink.text).strip()
    if tab_str.lower() == input_maincategory:
        onclick_str = ''
        try:
            onclick_str = str(tablink.get_attribute('onclick'))
            onclick_str = onclick_str.replace('openCity(event','').replace(')','').replace("/'",'').replace(',' ,'').strip()
            # print(onclick_str)
            onclick_str = removeSpecialCharacter(onclick_str)
        except:
            pass
        if onclick_str != '':
            try:
                tablink.click()
                sleep(5)
                subcat_str = 'cat2_xpath'
                subcat_str_path = config_dict[subcat_str.lower()]
                tabcontent_links = driver.find_elements_by_xpath(subcat_str_path)
                for tabcontent_link in tabcontent_links:
                    tabcontentid_str = str(tabcontent_link.get_attribute('id'))
                    # print(tabcontentid_str)
                    if tabcontentid_str == onclick_str:
                        category_heading = ''
                        try:
                            cat_head_str = 'subcat_head'
                            cat_head_str_path = config_dict[cat_head_str.lower()]
                            category_heading =  tabcontent_link.find_element_by_tag_name(cat_head_str_path).text
                        except:
                            pass
                        flag = True
                        count = 0
                        temp_product_link = ''
                        while flag == True:
                            prod_xpath = config_dict['prod_xpath']
                            active_links = tabcontent_link.find_elements_by_xpath(prod_xpath)
                            # print(len(active_links))
                            for i in range(0, len(active_links)):
                                if count == i:
                                    active_link = active_links[i]
                                    active_link_text = str(active_link.text).strip()
                                    if temp_product_link != active_link_text:
                                        # print(active_link_text)
                                        active_link.click()
                                        sleep(5)
                                        try:
                                            shades_container_path = config_dict['shade_container']
                                            # displayshades_container = driver.find_element_by_xpath('//div[@id="swatchDisplay"]')
                                            displayshades_container = driver.find_element_by_xpath(shades_container_path)
                                            # elementcontent = displayshades_container.get_attribute('outerHTML')
                                            # element_soup = BeautifulSoup(elementcontent, "html.parser")
                                            # print(element_soup)
                                            shades_links_path = config_dict['shade_xpath']
                                            shade_links = displayshades_container.find_elements_by_xpath(shades_links_path)
                                            # print(len(shade_links))
                                            shadecolor = ''
                                            shade_image = ''
                                            thumblinkimagename = ''
                                            shadecolor1 = ''
                                            for j in range(0, len(shade_links)):
                                                try:
                                                    if j != 0:
                                                        shade_link = shade_links[j]
                                                        shadelink_str = str(shade_link.text)
                                                        print(shadelink_str)
                                                        shade_link.click()
                                                        sleep(1)
                                                        # driver.execute_script("arguments[0].scrollIntoView();", category_element)
                                                        try:
                                                            # productname_element = driver.find_element_by_xpath('//h1[@class="product_title product-single__title"]')
                                                            shades_color_path = config_dict['shade_color']
                                                            shadecolor_element = driver.find_element_by_xpath(shades_color_path)
                                                            shadecolor = str(shadecolor_element.text).strip()
                                                        except:
                                                            pass
                                                        try:
                                                            shades_img_path = config_dict['shade_image']
                                                            shadeimglink_element = driver.find_element_by_xpath(shades_img_path)
                                                            thumblinkimage = str(shadeimglink_element.get_attribute('src'))
                                                            if shadecolor != '':
                                                                shadecolor1 = shadecolor
                                                                shadecolor = shadecolor.replace(' ', '_')
                                                                tab_str3 = removeSpecialCharacter(tab_str)
                                                                category_link_text3 = removeSpecialCharacter('')
                                                                active_link_text3 = removeSpecialCharacter(active_link_text)
                                                                shadelink_str3= removeSpecialCharacter(shadelink_str)
                                                                thumbimage_result_path = tab_str3 + '_' + category_link_text3 + '_' + active_link_text3 + '_' + shadelink_str3
                                                                output_images_folder = 'out_folder'
                                                                output_images_folder__path = config_dict[
                                                                    output_images_folder.lower()]
                                                                thumblinkimagename = downloadshadelink_image(thumblinkimage, thumbimage_result_path, output_images_folder__path)
                                                                # downloadshadelink_image(thumblinkimage, shadecolor)
                                                        except:
                                                            pass

                                                        driver.switch_to.frame(driver.find_element_by_tag_name('iframe'))
                                                        iframe_path = config_dict['iframe_path']
                                                        framecontent = driver.find_element_by_xpath(iframe_path)
                                                        # print(framecontent)
                                                        iframe_container = config_dict['iframe_container']
                                                        data_radiums = driver.find_elements_by_xpath(iframe_container)
                                                        # print(len(data_radiums))
                                                        data_radium_flag = False
                                                        for data_radium in data_radiums:
                                                            try:
                                                                iframe_images_path = config_dict['iframe_images']
                                                                imagetags = data_radium.find_elements_by_xpath(iframe_images_path)
                                                                # print(len(imagetags))
                                                                for k in range(0, len(imagetags)):
                                                                    if k == 4:
                                                                        largeimagepath = outputfile_path
                                                                        tab_str4 = removeSpecialCharacter(tab_str)
                                                                        category_link_texta4 = removeSpecialCharacter('')
                                                                        active_link_text4 = removeSpecialCharacter(active_link_text)
                                                                        shadelink_str4 = removeSpecialCharacter(shadelink_str)
                                                                        outputimagepath = tab_str4 + '_' + category_link_texta4 + '_' + active_link_text4 + '_' + shadelink_str4 + "_Large_image.jpg"
                                                                        # outputimagepath = "C:\\Solutions\\Lakmeindia\\output_images\\" + shadecolor + "_Large_image.jpg"
                                                                        temp = pd.DataFrame({'Category': tab_str,
                                                                                             'Sub Category': '',
                                                                                             'Product Name': active_link_text,
                                                                                             'Shade Name': shadelink_str,
                                                                                             'Shade Color': shadecolor1,
                                                                                             'ThumbImage Name': thumblinkimagename,
                                                                                             'LargeImage Name': largeimagepath + outputimagepath, },
                                                                                            index=[0])
                                                                        # print(tab_str)
                                                                        # print(category_link_text)
                                                                        # print(active_link_text)
                                                                        # print(shadelink_str)
                                                                        # print(shadecolor1)
                                                                        # print(thumblinkimagename)
                                                                        # print(outputimagepath)
                                                                        resultdata_df.append(temp)
                                                                        imagetag = imagetags[k]
                                                                        imagetag.click()
                                                                        sleep(10)
                                                                        output_images_folder = 'out_folder'
                                                                        output_images_folder__path = config_dict[output_images_folder.lower()]
                                                                        resultpath = output_images_folder__path
                                                                        try:
                                                                            os.rename(largeimagepath + 'ymk-beauty.jpg',largeimagepath + outputimagepath)
                                                                        except:
                                                                            pass
                                                                        try:
                                                                            shutil.copy(largeimagepath + outputimagepath,resultpath)
                                                                        except:
                                                                            pass
                                                                        try:
                                                                            os.remove(largeimagepath + outputimagepath)
                                                                        except:
                                                                            pass
                                                                        filelist = [f for f in os.listdir(largeimagepath) if f.endswith(".jpg")]
                                                                        for f in filelist:
                                                                            os.remove(os.path.join(largeimagepath, f))
                                                                        data_radium_flag = True
                                                                        break
                                                                if data_radium_flag == True:
                                                                    break
                                                            except:
                                                                pass
                                                        driver.switch_to.default_content()
                                                except:
                                                    continue
                                        except Exception as e:
                                            print(e)
                                            pass
                                        try:
                                            editlink_div = tabcontent_link.find_element_by_xpath(
                                                ".//div[contains(@class,'editCat')]")
                                            editlink = editlink_div.find_element_by_tag_name("a").click()
                                            sleep(5)
                                            flag = True
                                            count += 1
                                            break
                                        except:
                                            flag = False
                                            break
                                    temp_product_link = active_link_text
                            if count == len(active_links):
                                flag = False
                                break
            except:
                continue

        break
if len(resultdata_df) > 0:
    if len(resultdata_df) > 0:
        resultdata_df = pd.concat(resultdata_df, ignore_index=True)
        resultdata_df.insert(loc=0, column='S.No', value=np.arange(1, len(resultdata_df) + 1))
        resultdata_df.to_csv(outputfile, encoding='utf-8-sig', index=False)
    endtime = datetime.now()
    diff = endtime - starttime
    print(diff.seconds)