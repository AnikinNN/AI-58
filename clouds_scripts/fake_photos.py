import os

photos_fake_path = r"E:\AI-58\clouds_photos_fake"
photos_path = r"E:\AI-58\clouds_photos"

for directory in os.listdir(photos_path):
    directory_path = os.path.join(photos_path, directory)
    if os.path.isdir(directory_path):
        try:
            os.mkdir(os.path.join(photos_fake_path, directory))
        except FileExistsError:
            pass

        for photo in os.listdir(directory_path):
            with open(os.path.join(photos_fake_path, directory, photo), "w"):
                pass
