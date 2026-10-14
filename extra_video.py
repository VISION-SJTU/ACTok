import cv2
import os
import sys

def save_video_frames(video_path, output_dir):
    """
    将视频的每一帧保存为图像文件
    
    参数:
        video_path: 输入视频文件路径
        output_dir: 输出目录路径
    """
    
    # 检查视频文件是否存在
    if not os.path.exists(video_path):
        print(f"错误: 视频文件 '{video_path}' 不存在")
        return False
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    # 检查视频是否成功打开
    if not cap.isOpened():
        print("错误: 无法打开视频文件")
        return False
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"视频信息:")
    print(f"  - 文件: {video_path}")
    print(f"  - 帧率: {fps:.2f} FPS")
    print(f"  - 总帧数: {total_frames}")
    print(f"  - 时长: {duration:.2f} 秒")
    print(f"  - 输出目录: {output_dir}")
    print("开始提取帧...")
    
    frame_count = 0
    success = True
    
    # 逐帧读取并保存
    while success:
        success, frame = cap.read()
        
        if success:
            # 生成文件名，用6位数字补零 (例如: frame_000001.jpg)
            filename = f"frame_{frame_count:06d}.jpg"
            filepath = os.path.join(output_dir, filename)
            
            # 保存帧为JPEG图像
            cv2.imwrite(filepath, frame)
            
            # 每50帧显示一次进度
            if frame_count % 50 == 0:
                print(f"已处理 {frame_count}/{total_frames} 帧...")
            
            frame_count += 1
    
    # 释放视频捕获对象
    cap.release()
    
    print(f"\n完成! 成功保存 {frame_count} 帧图像到 '{output_dir}' 目录")
    print(f"图像格式: JPEG, 命名: frame_000001.jpg, frame_000002.jpg, ...")
    
    return True

def main():
    # 设置视频路径和输出目录
    video_file = "cats.mov"
    output_directory = "cats"
    
    # 执行帧提取
    success = save_video_frames(video_file, output_directory)
    
    if not success:
        print("帧提取失败")
        sys.exit(1)

if __name__ == "__main__":
    main()