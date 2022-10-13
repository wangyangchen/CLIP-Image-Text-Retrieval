import os
img_name_file = "E:/project/clip/CLIP-main/data/Mscoco/testall.txt"
# text_file = "D:/Adata/scan_data/coco_precomp/dev_caps.txt"
new_file = "E:/project/clip/CLIP-main/data/Mscoco/testall_local.txt"

train_data_path = "E:/data/MSCOCO2017/train2017/"
val_data_path = "E:/data/MSCOCO2017/val2017/"

imgs_list = []
with open(img_name_file, "r") as f:
    for line in f:
        if os.path.exists((train_data_path + line).replace("\n", "")):
            imgname = (train_data_path + line).replace("\n", "")
            imgs_list.append(imgname+"\n")
        elif os.path.exists((val_data_path + line).replace("\n", "")):
            imgname = (val_data_path + line).replace("\n", "")
            imgs_list.append(imgname+"\n")
image_filenames = imgs_list
print(len(image_filenames))
with open(new_file, "a") as f:
    f.writelines(image_filenames)



# sentences_list = []
# with open(text_file, "r") as f:
#     for line in f:
#         sentences_list.append(line.replace("\n", ""))
# text_filenames = sentences_list