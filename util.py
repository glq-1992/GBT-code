import os


def check_path():
    txt_f = 'DatasetFile/Split_template/T_test.txt'
    with open(txt_f) as f:
        for line in f.readlines():
            path,_ = line.split('Q')
            if(os.path.exists(path) ==1):
                pass
            else:
                print(path)
def check_data():
    txt_file = '/home/gaoliqing/code_glq/ICCV_2021/VAC_CSLR-main/DatasetFile/Split_insert_com/140replace_dev.txt'
    with open(txt_file) as f:
        for line in f.readlines():
            video_path, _ = line.split('Q')
            if not os.path.exists(video_path):
                print(video_path)
            # elif not os.path.exists(video_path1):
            #     print(video_path1)
def add_path():
    read_file = 'DatasetFile/Split_maskVAC/response_tr.txt'
    mask_prefix = '/hd3/dataset/CSL-TJU_mask_112/Response'
    save_file = 'DatasetFile/Split_maskVAC/response_train.txt'
    new_lines = []
    with open(read_file) as f:
        for line in f.readlines():
            img_pth, label = line.split('Q')
            folder = img_pth[-18:]
            mask_pth = mask_prefix + folder
            new_line = img_pth+'Q'+mask_pth+'Q'+label
            new_lines.append(new_line)
    with open(save_file,'w') as f:
        for line in new_lines:
            f.write(line)

# def check_data():
#     txt_path = 'DatasetFile/Split_TER/insert_dev.txt'
#     with open(txt_path) as f:
#         for line in f.readlines():
#             file_path, _ = line.split('Q')
#             length = len(os.listdir(file_path))
#             if length == 0:
#                 print(file_path)
def temp():
    f1 = 'R3003.txt'
    f2 = 'R3004.txt'
    f1_list = []
    with open(f2) as f_1:
        for line in f_1.readlines():
            f1_list.append(line)
    with open(f1) as f_2:
        for line in f_2.readlines():
            if line not in f1_list:
                print(line)

if __name__ == '__main__':
    check_data()