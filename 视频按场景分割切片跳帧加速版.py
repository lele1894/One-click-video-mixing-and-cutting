import os
import time
import subprocess
import threading
from tkinter import Tk, Label, Button, filedialog, StringVar, DoubleVar, messagebox, Scale, Listbox, END, Scrollbar, RIGHT, Y
import ffmpeg
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector

# 日志工具，用于更新日志框
def log(message):
    log_list.insert(END, f"[日志] {message}")
    log_list.yview(END)  # 滚动到最新日志

# 装饰器：用于计算函数运行时间
def time_it(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        log(f"{func.__name__} 执行耗时: {elapsed_time:.2f} 秒")
        return result
    return wrapper

# 使用 FFmpeg 获取视频基本信息
def get_video_info(video_path):
    try:
        probe = ffmpeg.probe(video_path)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        audio_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)

        # 获取视频信息
        if video_stream:
            bitrate = int(probe['format']['bit_rate']) // 1000  # 转换为 kbps
            fps = eval(video_stream['avg_frame_rate'])  # 计算帧率
            resolution = f"{video_stream['width']}x{video_stream['height']}"

        # 获取音频信息
        audio_info = f"{audio_stream['codec_name']} {audio_stream['sample_rate']}Hz" if audio_stream else "无音频信息"

        return bitrate, fps, resolution, audio_info
    except Exception as e:
        log(f"无法获取视频信息: {e}")
        return None, None, None, None

# 使用 FFmpeg 根据场景列表分割视频
def split_video(video_path, scene_list, output_dir):
    os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在

    # 根据场景列表生成分割时间
    segment_times = ",".join(str(scene[0].get_seconds()) for scene in scene_list)
    output_template = os.path.join(output_dir, "output_%03d.mp4")

    # 构造 FFmpeg 命令
    ffmpeg_cmd = [
        "ffmpeg", "-i", video_path,
        "-c", "copy",  # 不重新编码
        "-f", "segment",  # 按片段输出
        "-segment_times", segment_times,
        "-reset_timestamps", "1",  # 重置时间戳
        "-avoid_negative_ts", "make_zero",  # 避免负时间戳
        output_template
    ]

    log(f"执行命令: {' '.join(ffmpeg_cmd)}")
    subprocess.run(ffmpeg_cmd, check=True)

# 检测视频场景并减少帧处理
@time_it
def detect_scenes(video_path, threshold, frame_skip=5):
    """
    检测场景，减少帧处理通过设置采样间隔。
    ####################################################
    :param video_path: 视频文件路径
    :param threshold: 场景检测阈值
    :param frame_skip: 跳过的帧数，每 frame_skip+1 帧处理一次
    :return: 检测到的场景列表
    """
    try:
        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=threshold))

        # 只对采样帧进行场景检测
        scene_manager.detect_scenes(
            frame_source=video,
            frame_skip=frame_skip  # 跳过帧数
        )

        scene_list = scene_manager.get_scene_list()
        log(f"检测到 {len(scene_list)} 个场景。")
        return scene_list
    except Exception as e:
        log(f"场景检测失败: {e}")
        return []


# 处理视频：检测场景并分割
@time_it
def detect_and_split_video(video_path, threshold, frame_skip=5):
    log("开始场景检测...")
    # 增加 frame_skip 参数控制帧采样
    scene_list = detect_scenes(video_path, threshold, frame_skip=frame_skip)

    if not scene_list:
        log("未检测到场景，跳过视频分割。")
        return

    output_dir = f"{os.path.splitext(video_path)[0]}_{int(threshold)}"
    log(f"分割视频将存储到目录: {output_dir}")

    log("开始视频分割...")
    split_video(video_path, scene_list, output_dir)

    output_count = len([f for f in os.listdir(output_dir) if f.endswith(".mp4")])
    log(f"分割完成，共输出 {output_count} 个文件。")


# 在子线程中运行视频处理，避免阻塞主线程
def process_video_in_thread():
    if not video_path.get():
        log("请先选择视频文件！")
        return

    try:
        threshold = detection_threshold.get()
        threading.Thread(target=detect_and_split_video, args=(video_path.get(), threshold), daemon=True).start()
    except Exception as e:
        log(f"处理过程中发生错误: {e}")

# GUI 主界面
def select_video():
    path = filedialog.askopenfilename(title="选择视频文件", filetypes=[("视频文件", "*.mp4")])
    if path:
        video_path.set(path)
        log(f"选择的视频: {video_path.get()}")

        # 显示视频基本信息
        bitrate, fps, resolution, audio_info = get_video_info(video_path.get())
        log(f"视频信息 - 码率: {bitrate} kbps, 帧率: {fps:.2f} FPS, 分辨率: {resolution}, 音频: {audio_info}")

# 主程序入口
if __name__ == "__main__":
    # 初始化 Tkinter 界面
    root = Tk()
    root.title("视频场景分割工具")
    root.geometry("700x500")

    # 定义变量
    video_path = StringVar()
    detection_threshold = DoubleVar(value=30.0)  # 默认阈值为 30.0

    # 界面布局
    Label(root, text="选择视频文件:").pack(pady=10)
    Button(root, text="选择视频", command=select_video).pack(pady=5)
    Label(root, textvariable=video_path, wraplength=600).pack()

    Label(root, text="场景检测阈值 (5-95):").pack(pady=10)
    Scale(root, from_=5.0, to=95.0, resolution=5.0, orient="horizontal", variable=detection_threshold).pack()

    Button(root, text="开始处理", command=process_video_in_thread, bg="green", fg="white").pack(pady=20)

    # 日志显示框
    log_list = Listbox(root, height=15)
    log_list.pack(fill="both", expand=True, padx=10, pady=10)
    scrollbar = Scrollbar(log_list)
    scrollbar.pack(side=RIGHT, fill=Y)
    log_list.config(yscrollcommand=scrollbar.set)
    scrollbar.config(command=log_list.yview)

    # 启动主循环
    root.mainloop()
