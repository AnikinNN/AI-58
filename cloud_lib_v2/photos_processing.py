import os
import re


def get_photo_names(photos_path):
    photos = []
    # get list of photos names
    if os.path.isdir(photos_path):
        # there can be two situations:
        #     - photos_path contains photos: "img-2021-07-26T15-25-46devID1.jpg"
        #     - photos_path contains subdirectories: "snapshots-2021-07-26"
        photos_path_content = os.listdir(photos_path)
        for content in photos_path_content:
            if re.findall(r"img-[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}-[0-9]{2}-[0-9]{2}devID[0-9]+.jpg", content):
                photos.append(content)
            elif re.findall(r"snapshots-[0-9]{4}-[0-9]{2}-[0-9]{2}\Z", content):
                photos.extend(os.listdir(os.path.join(photos_path, content)))
    return photos


def get_full_path(photo_name, photos_base_dir):
    return os.path.join(photos_base_dir, "snapshots-" + extract_time(photo_name)[:10], photo_name)


def extract_time(file_name):
    """
    extracts time and date from file_name
    :param file_name: string, example "img-2021-08-07T18-02-19devID1.jpg"
    :return: string, example "2021-08-07 18:02:19"
    """
    result = re.findall("[0-9]{4}-[0-9]{2}-[0-9]{2}", file_name)[0]
    result += " "
    result += re.findall("T[0-9]{2}-[0-9]{2}-[0-9]{2}", file_name)[0][1:].replace("-", ":")
    return result
