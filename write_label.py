# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import random

signal = 0
folder_name = []
image_path = []
xml_path = []

for directory, folders, files in os.walk('/data1/object/DATA'):
    if signal == 0:
        folder_name.extend(folders)
	folder_name.sort()
    
    if signal > 0:
        for file in files:
            file_format = os.path.splitext(file)[1].strip('.')
            if file_format in ['jpg','JPG','jpeg','JPEG']:
                image_path.append(os.path.join(directory,file))
            elif file_format in ['xml','XML']:
                xml_path.append(os.path.join(directory,file))
            
    signal += 1
    if signal % 10 == 0:
        print(signal)
        
xml_random_index = [i for i in range(len(xml_path))]
random.shuffle(xml_random_index)

xml_random_list = []
for index in xml_random_index:
    xml_random_list.append(xml_path[index])
    
with open('DATA/xml_random.txt','w') as f:
    for index, item in enumerate(xml_random_list):
        if index < len(xml_random_list) - 1:
            f.write('{}\n'.format(item))
        else:
            f.write('{}'.format(item))

write_folder_name = []
#write_folder_name.append('none_of_the_above')
for i in folder_name:
    write_folder_name.append(i)
            
with open('DATA/object_label.pbtxt','w') as f:
    for index, item in enumerate(write_folder_name):
        if index < len(write_folder_name) - 1:
            f.write('item {\n  id: %d\n  name: \'%s\'\n}\n\n' % (index+1,item))
        else:
            f.write('item {\n  id: %d\n  name: \'%s\'\n}' % (index+1,item))
    
        
