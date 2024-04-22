import shutil
import os
import zipfile

def main():
    
    DIR = "/home/sanghyeon/Downloads/3obj_prefV+prefpolicy_tree_reinv"
    zip_files = os.listdir(DIR)

    for fname in zip_files:
        with zipfile.ZipFile(os.path.join(DIR, fname)) as zip_ref:
            zip_ref.extractall(DIR)





if __name__=="__main__":
    main()