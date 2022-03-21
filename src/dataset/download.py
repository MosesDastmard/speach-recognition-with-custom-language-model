import os
from src.util.config import COMMON_VOICE_URL, COMMON_VOICE_TAR_PATH, DATA_PATH
from libs.datil.flag import Flag
from src.util.functions import Process



class Download:
    def __init__(self, url, file_path=None) -> None:
        self.url = url
        self.file_path = file_path

    def run(self):
        if self.file_path is None:
            download_command = f"wget -t 5 -c {self.url}"
        else:
            download_command = f"wget -t 5 -O {self.file_path} -c {self.url}"
        Process(download_command).run()


class Extract:
    def __init__(self, file_path, destination_path=None) -> None:
        self.file_path = file_path
        self.destination_path = destination_path

    def run(self):
        flag = Flag('extract')
        file_extention = flag.get_file_info(self.file_path)['extention']
        if flag.exists(self.file_path):
            print(f'the file {self.file_path} is already extracted.')
        else:
            if file_extention == ".gz":
                if self.destination_path is None:
                    extract_command = f'tar -xvf {self.file_path}'
                else:
                    os.makedirs(self.destination_path, exist_ok=True)
                    extract_command = f'tar -xvf {self.file_path} -C {self.destination_path}'
            elif file_extention == ".xz":
                extract_command = f'xz -d -v {self.file_path}'
            Process(extract_command).run()
            flag.put(self.file_path)