import os
import requests
import tarfile
from tqdm import tqdm

class ImageNetDownloader:
    def __init__(self, username, access_key, download_dir="./imagenet"):
        self.username = username
        self.access_key = access_key
        self.download_dir = download_dir
        self.base_url = "http://www.image-net.org/challenges/LSVRC/2012/dd31405981ef5f776aa17412e1f0c112"
        
        os.makedirs(download_dir, exist_ok=True)
    
    def download_file(self, url, filename):
        """下载文件并显示进度条"""
        local_filename = os.path.join(self.download_dir, filename)
        
        # 如果文件已存在，跳过下载
        if os.path.exists(local_filename):
            print(f"{filename} 已存在，跳过下载")
            return local_filename
        
        # 设置认证信息
        auth = (self.username, self.access_key)
        
        # 发起请求
        response = requests.get(url, auth=auth, stream=True)
        response.raise_for_status()
        
        # 获取文件总大小
        total_size = int(response.headers.get('content-length', 0))
        
        # 下载文件
        with open(local_filename, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)
        
        return local_filename
    
    def download_imagenet(self):
        """下载ImageNet数据集"""
        files = {
            "ILSVRC2012_img_train.tar": f"{self.base_url}/ILSVRC2012_img_train.tar",
            "ILSVRC2012_img_val.tar": f"{self.base_url}/ILSVRC2012_img_val.tar",
            "ILSVRC2012_img_test_v10102019.tar": f"{self.base_url}/ILSVRC2012_img_test_v10102019.tar",
            "ILSVRC2012_devkit_t12.tar.gz": f"{self.base_url}/ILSVRC2012_devkit_t12.tar.gz"
        }
        
        for filename, url in files.items():
            print(f"开始下载 {filename}...")
            self.download_file(url, filename)
            print(f"下载完成: {filename}")

# 使用示例
if __name__ == "__main__":
    # 需要替换为你的用户名和访问密钥
    downloader = ImageNetDownloader(
        username="你的用户名",
        access_key="你的访问密钥"
    )
    downloader.download_imagenet()