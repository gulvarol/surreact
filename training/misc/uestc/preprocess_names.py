"""
Naming convention of original videos were not consistent for the subject_id part,
so some were renamed. See `preprocess_names.sh` as the printed output of this script.
"""
import shutil
import os


cnt_edit = 0
cnt_fine = 0
data_root = "/home/gvarol/datasets/hri40/RGBvideo/"
for filename in os.listdir(data_root):
    if filename.endswith("avi"):
        subject_str = filename.split("_")[2]
        if len(subject_str) == 3:
            cnt_edit = cnt_edit + 1
            subject_id = int(subject_str[1:])
            subject_str_new = "p{:03d}".format(subject_id)
            filename_new = filename.replace(subject_str, subject_str_new)
            fullfile = os.path.join(data_root, filename)
            fullfile_new = os.path.join(data_root, filename_new)
            if os.path.isfile(fullfile_new):
                print(filename)
                print("1. This should not happen.")
                exit()
            else:
                print("mv {} {}".format(fullfile, fullfile_new))
                # shutil.move(fullfile, fullfile_new)
        elif len(subject_str) == 4:
            cnt_fine = cnt_fine + 1
        else:
            print(filename)
            print("2. This should not happen.")
            exit()

print("Changed {} file names.".format(cnt_edit))
print("{} files were fine.".format(cnt_fine))
