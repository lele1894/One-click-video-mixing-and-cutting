import os
import time
import subprocess
import threading
from tkinter import Tk, Label, Button, filedialog, StringVar, DoubleVar, messagebox, Scale, Listbox, END, Scrollbar, RIGHT, Y, BooleanVar, Checkbutton, Frame, LabelFrame, LEFT
import random
from pathlib import Path
import json
import hashlib
import ffmpeg
from scenedetect import detect, AdaptiveDetector, split_video_ffmpeg, SceneManager, open_video, ContentDetector
from scenedetect.scene_manager import save_images, write_scene_list
from scenedetect.stats_manager import StatsManager
from scenedetect.video_manager import VideoManager
from scenedetect.frame_timecode import FrameTimecode
import sys
import cv2

# 全局变量声明
def init_globals(root_window):
    """初始化全局变量"""
    global log_list, video_path, detection_threshold
    log_list = None
    video_path = StringVar(root_window)
    detection_threshold = DoubleVar(root_window, value=30.0)

# 检查 FFmpeg 是否已安装
def check_ffmpeg():
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        messagebox.showerror("错误", 
            "未找到 FFmpeg!\n\n"
            "请安装 FFmpeg 并确保已添加到系统环境变量。\n"
            "Windows 用户可以访问: https://ffmpeg.org/download.html\n"
            "下载后将 ffmpeg.exe 所在目录添加到系统环境变量 Path 中。"
        )
        return False

# 日志工具，用于更新日志框
def log(message):
    """日志工具，用于更新日志框"""
    if log_list is not None:
        log_list.insert(END, f"[日志] {message}")
        log_list.yview(END)  # 滚动到最新日志
    else:
        print(f"[日志] {message}")  # 如果log_list未初始化，则打印到控制台

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
            
            # 获取视频时长（秒）
            duration = float(probe['format']['duration'])
            # 转换为时:分:秒格式
            hours = int(duration // 3600)
            minutes = int((duration % 3600) // 60)
            seconds = int(duration % 60)
            duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        # 获取音频信息
        audio_info = f"{audio_stream['codec_name']} {audio_stream['sample_rate']}Hz" if audio_stream else "无音频信息"

        return bitrate, fps, resolution, audio_info, duration_str
    except Exception as e:
        log(f"无法获取视频信息: {e}")
        return None, None, None, None, None

# 修改音频提取函数
def extract_audio(video_path, output_dir):
    """提取视频中的音频"""
    # 获取原始视频的完整文件名（包含扩展名）
    video_basename = os.path.basename(video_path)
    # 音频文件保存到原始视频所在目录
    audio_path = os.path.join(os.path.dirname(video_path), f"{video_basename}_original_audio.aac")
    
    try:
        ffmpeg_cmd = [
            "ffmpeg", "-i", video_path,
            "-vn", "-acodec", "copy", "-y",  # 不转码，直接复制音频流
            audio_path
        ]
        subprocess.run(ffmpeg_cmd, check=True)
        log(f"音频提取完成: {audio_path}")
        return audio_path
    except Exception as e:
        log(f"音频提取失败: {e}")
        return None

# 修改split_video函，添加无音频选项
def split_video(video_path, scene_list, output_dir, include_audio=True, merge_random=False):
    """分割视频
    :param video_path: 视频路径
    :param scene_list: 场景列表
    :param output_dir: 输出目录
    :param include_audio: 是否包含音频
    :param merge_random: 是否需要随机合并
    """
    os.makedirs(output_dir, exist_ok=True)
    
    segment_times = ",".join(str(scene.get_start()) for scene in scene_list)
    output_template = os.path.join(output_dir, "output_%03d.mp4")
    
    # 基础命令
    ffmpeg_cmd = [
        "ffmpeg", "-i", video_path,
        "-f", "segment",
        "-segment_times", segment_times,
        "-reset_timestamps", "1",
        "-avoid_negative_ts", "make_zero",
    ]
    
    if include_audio and not merge_random:
        # 只有在不需要随机合并时才保留音频
        ffmpeg_cmd.extend(["-c", "copy"])
    else:
        # 其他情况都去除音频
        ffmpeg_cmd.extend(["-an", "-c:v", "copy"])
        
    ffmpeg_cmd.append(output_template)
    
    log(f"执行命令: {' '.join(ffmpeg_cmd)}")
    subprocess.run(ffmpeg_cmd, check=True)

# 修改随机合并函数
def merge_random_clips(input_dir, audio_path=None, keep_temp=False):
    """随机合并视频片段并加原始音频"""
    clips = [f for f in os.listdir(input_dir) if f.startswith("output_") and f.endswith(".mp4")]
    random.shuffle(clips)
    
    # 创建合并列表文件
    list_file = os.path.join(input_dir, "concat_list.txt")
    with open(list_file, "w", encoding="utf-8") as f:
        for clip in clips:
            f.write(f"file '{os.path.join(input_dir, clip)}'\n")
    
    # 获取原始视频路径信息
    dir_name = os.path.basename(input_dir)
    original_name = "_".join(dir_name.split("_")[:-1])
    
    # 在原始视频所在目录中查找匹配的视频文件
    video_dir = os.path.dirname(input_dir)
    video_files = [f for f in os.listdir(video_dir) if f.startswith(original_name) and f.endswith((".mp4", ".mkv", ".avi"))]
    
    if video_files:
        original_video_name = video_files[0]
        base_name = os.path.splitext(original_video_name)[0]
    else:
        base_name = original_name
    
    # 修改输出文件命名
    output_path = os.path.join(video_dir, f"{base_name}_s.mp4")
    temp_output = os.path.join(input_dir, "temp_merged.mp4")
    
    try:
        # 合并视频片段
        subprocess.run([
            "ffmpeg", "-f", "concat", "-safe", "0",
            "-i", list_file,
            "-c", "copy", "-y",
            temp_output
        ], check=True)
        
        if audio_path:
            # 添加原始音频
            subprocess.run([
                "ffmpeg", "-i", temp_output,
                "-i", audio_path,
                "-c", "copy",
                "-map", "0:v:0", "-map", "1:a:0", "-y",
                output_path
            ], check=True)
        else:
            os.replace(temp_output, output_path)
            
        log(f"随机合并完成: {output_path}")
    except Exception as e:
        log(f"合并过程出错: {e}")
    finally:
        # 清理临时文件
        if not keep_temp:
            # 删除临时文件
            if os.path.exists(temp_output):
                os.remove(temp_output)
            if os.path.exists(list_file):
                os.remove(list_file)
            if os.path.exists(audio_path):
                os.remove(audio_path)
            # 删除切片文件夹
            for clip in clips:
                clip_path = os.path.join(input_dir, clip)
                if os.path.exists(clip_path):
                    os.remove(clip_path)
            try:
                os.rmdir(input_dir)
            except:
                pass

# 添加计算视频文件 MD5 的函数
def calculate_video_md5(file_path, block_size=8192):
    """计算视频文件的 MD5 值，只读取前 1MB 数据以提高速度"""
    md5 = hashlib.md5()
    max_size = 1024 * 1024  # 1MB
    total_read = 0
    
    with open(file_path, 'rb') as f:
        while True:
            data = f.read(block_size)
            if not data or total_read >= max_size:
                break
            md5.update(data)
            total_read += len(data)
    return md5.hexdigest()

# 添加场景数据保存和加载函数
def get_cache_path(video_path, threshold):
    """获取缓存文件路径"""
    video_md5 = calculate_video_md5(video_path)
    cache_dir = os.path.join(os.path.dirname(video_path), '.scene_cache')
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{video_md5}_{int(threshold)}.json")

def save_scene_data(video_path, threshold, scene_list):
    """保存场景检测数据"""
    cache_path = get_cache_path(video_path, threshold)
    
    # 修改场景数据的保存格式
    scene_data = {
        'video_path': video_path,
        'threshold': threshold,
        'scenes': [[scene.get_start(), scene.get_end()] for scene in scene_list],
        'timestamp': time.time()
    }
    
    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(scene_data, f, indent=2)
        log(f"场景数据已缓存: {cache_path}")
    except Exception as e:
        log(f"保存场景数据失败: {e}")

def load_scene_data(video_path, threshold):
    """加载场景检测数据"""
    cache_path = get_cache_path(video_path, threshold)
    
    if not os.path.exists(cache_path):
        return None
        
    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 检查数据是否过期（可选，这里设置为30天）
        if time.time() - data['timestamp'] > 30 * 24 * 3600:
            log("缓存数据已过期")
            return None
            
        log(f"已加载缓存的场景数据: {cache_path}")
        return data['scenes']
    except Exception as e:
        log(f"加载场景数据失败: {e}")
        return None

# 修改 Scene 类的定义
class Scene:
    def __init__(self, start_time, end_time):
        self.start_time = start_time
        self.end_time = end_time
    
    def get_start(self):
        """返回场景开始时间"""
        return self.start_time
    
    def get_end(self):
        """返回场景结束时间"""
        return self.end_time

# 修改 detect_scenes 函数
def detect_scenes(video_path, threshold, frame_skip=5):
    """检测场景，如果有缓存则直接加载"""
    # 先尝试加载缓存数据
    cached_scenes = load_scene_data(video_path, threshold)
    if cached_scenes is not None:
        log("使用缓存的场景数据")
        # 将缓存数据转换回场景列表格式
        scene_list = []
        for start, end in cached_scenes:
            scene = Scene(start, end)
            scene_list.append(scene)
        return scene_list
    
    # 如果没有缓存，执行检测
    try:
        # 检查是否有可用的 CUDA 设备
        cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
        if cuda_available:
            log("使用 NVIDIA GPU 加速场景检测")
            # 创建 CUDA Stream
            stream = cv2.cuda_Stream()
            
        # 使用 ContentDetector 进行场景检测
        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(
            ContentDetector(
                threshold=threshold,
                # 如果有 GPU 则使用 CUDA
                gpu_enabled=cuda_available
            )
        )

        # 只对采样帧进行场景检测
        scene_manager.detect_scenes(
            frame_source=video,
            frame_skip=frame_skip  # 跳过帧数
        )

        # 获取场景列表
        detected_scenes = scene_manager.get_scene_list()
        
        # 转换为我们的场景对象格式
        scene_list = []
        for scene in detected_scenes:
            start_time = scene[0].get_seconds()
            end_time = scene[1].get_seconds()
            scene_list.append(Scene(start_time, end_time))

        log(f"检测到 {len(scene_list)} 个场景")
        
        # 保存检测结果
        save_scene_data(video_path, threshold, scene_list)
        
        return scene_list
    except Exception as e:
        log(f"场景检测失败: {e}")
        return []

# 添加获取视频帧率的辅助函数
def get_video_fps(video_path):
    """获取视频帧率"""
    try:
        probe = ffmpeg.probe(video_path)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        return eval(video_info['avg_frame_rate'])
    except Exception:
        return 30.0  # 默认帧率

def format_time(seconds):
    """格式化时间为 HH:MM:SS 格式"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

# 修改 detect_and_split_video 函数
def detect_and_split_video(video_path, threshold, frame_skip=5, keep_temp=False):
    """检测场景并分割视频（保留音频）"""
    # 场景检测计时
    start_detect = time.time()
    log("开始场景检测...")
    scene_list = detect_scenes(video_path, threshold, frame_skip=frame_skip)
    detect_time = time.time() - start_detect
    log(f"场景检测耗时: {detect_time:.2f} 秒")
    
    if not scene_list:
        log("未检测到场景，跳过视频分割。")
        return
        
    # 创建输出目录
    output_dir = f"{os.path.splitext(video_path)[0]}_{int(threshold)}"
    os.makedirs(output_dir, exist_ok=True)
    log("开始分割视频...")
    
    # 分割计时
    start_split = time.time()
    
    # 准备分割时间点
    segment_times = ",".join(str(scene.get_start()) for scene in scene_list[1:])  # 跳过第一个场景的开始时间
    output_template = os.path.join(output_dir, "scene_%03d.mp4")
    
    try:
        # 使用 FFmpeg segment 模式一次性分割所有场景
        subprocess.run([
            'ffmpeg', '-hide_banner', '-loglevel', 'error',
            '-i', video_path,
            '-f', 'segment',
            '-segment_times', segment_times,
            '-reset_timestamps', '1',
            '-c', 'copy',  # 直接复制流，不重新编码
            '-avoid_negative_ts', 'make_zero',
            '-y',
            output_template
        ], check=True, capture_output=True)
        
        # 统计输出文件数量
        output_count = len([f for f in os.listdir(output_dir) if f.endswith('.mp4')])
        split_time = time.time() - start_split
        
        # 最终统计
        log(f"分割完成: 输出 {output_count} 个场景")
        log(f"总耗时: {split_time:.2f} 秒")
        
        # 打开输出目录
        try:
            if os.name == 'nt':  # Windows
                os.startfile(output_dir)
            else:  # Linux/Mac
                subprocess.run(['xdg-open', output_dir])
        except Exception as e:
            log(f"无法打开输出目录: {e}")
            
    except subprocess.CalledProcessError as e:
        log(f"分割失败: {e.stderr.decode()}")

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
        bitrate, fps, resolution, audio_info, duration = get_video_info(video_path.get())
        log(f"视频信息 - 时长: {duration}, 码率: {bitrate} kbps, 帧率: {fps:.2f} FPS, 分辨率: {resolution}, 音频: {audio_info}")

# 添加 start_processing 函数
def start_processing(keep_temp):
    if not video_path.get():
        log("请先选择视频文件！")
        return
        
    try:
        threshold = detection_threshold.get()
        threading.Thread(
            target=detect_and_split_video,
            args=(
                video_path.get(),
                threshold,
                5,  # frame_skip
                keep_temp
            ),
            daemon=True
        ).start()
    except Exception as e:
        log(f"处理过程中发生错误: {e}")

# 添加文件夹选择功能
def select_folder():
    """选择视频文件夹"""
    folder_path = filedialog.askdirectory(title="选择视频文件夹")
    if folder_path:
        # 获取所有视频文件
        video_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(('.mp4', '.mkv', '.avi'))]
        if not video_files:
            messagebox.showwarning("警告", "所选文件夹中没有视频文件！")
            return
            
        # 按文件名排序
        video_files.sort()
        
        # 显示找到的视频文件
        log(f"找到 {len(video_files)} 个视频文件:")
        for file in video_files:
            log(f"- {file}")
        
        # 生成输出文件名
        output_file = os.path.join(folder_path, "merged.mp4")
        
        # 如果文件已存在，添加数字后缀
        counter = 1
        while os.path.exists(output_file):
            base_name = "merged"
            output_file = os.path.join(folder_path, f"{base_name}_{counter}.mp4")
            counter += 1
        
        # 在新线程中执行合并
        threading.Thread(
            target=merge_folder_videos,
            args=(folder_path, video_files, output_file),
            daemon=True
        ).start()

def merge_folder_videos(folder_path, video_files, output_file):
    """合并文件夹中的视频文件"""
    try:
        # 创建合并列表文件
        list_file = os.path.join(folder_path, 'concat_list.txt')
        with open(list_file, 'w', encoding='utf-8') as f:
            for video in video_files:
                # 使用绝对路径并转换为正斜杠
                video_path = os.path.join(folder_path, video)
                abs_path = os.path.abspath(video_path).replace('\\', '/')
                f.write(f"file '{abs_path}'\n")
        
        log("开始合并视频...")
        start_time = time.time()
        
        # 执行合并
        subprocess.run([
            'ffmpeg', '-hide_banner', '-loglevel', 'error',
            '-f', 'concat',
            '-safe', '0',
            '-i', list_file,
            '-c', 'copy',
            '-y', output_file
        ], check=True)
        
        merge_time = time.time() - start_time
        log(f"合并完成: {os.path.basename(output_file)}")
        log(f"合并耗时: {merge_time:.2f} 秒")
        
        # 清理临时文件
        os.remove(list_file)
        
        # 打开输出文件所在目录
        try:
            if os.name == 'nt':  # Windows
                os.startfile(folder_path)
            else:  # Linux/Mac
                subprocess.run(['xdg-open', folder_path])
        except Exception as e:
            log(f"无法打开输出目录: {e}")
            
    except subprocess.CalledProcessError as e:
        log(f"合并失败: {e.stderr.decode() if hasattr(e, 'stderr') else str(e)}")
    except Exception as e:
        log(f"合并过程出错: {e}")
    finally:
        if os.path.exists(list_file):
            os.remove(list_file)

# 修改 GUI 布局，添加文件夹选择按钮
def create_gui(root):
    """创建GUI界面"""
    global log_list
    
    # 创建主框架
    main_frame = Frame(root, padx=20, pady=10)
    main_frame.pack(fill="both", expand=True)
    
    # 文件选择区域
    file_frame = LabelFrame(main_frame, text="文件选择", padx=10, pady=5)
    file_frame.pack(fill="x", pady=(0, 10))
    
    # 添加按钮框架
    button_frame = Frame(file_frame)
    button_frame.pack(pady=5)
    
    Button(button_frame, text="选择视频", command=select_video, width=15).pack(side=LEFT, padx=5)
    Button(button_frame, text="选择文件夹", command=select_folder, width=15).pack(side=LEFT, padx=5)
    
    Label(file_frame, textvariable=video_path, wraplength=600).pack()
    
    # 参数设置区域
    settings_frame = LabelFrame(main_frame, text="参数设置", padx=10, pady=5)
    settings_frame.pack(fill="x", pady=(0, 10))
    
    # 阈值设置
    threshold_frame = Frame(settings_frame)
    threshold_frame.pack(fill="x", pady=5)
    Label(threshold_frame, text="场景检测阈值:").pack(side=LEFT)
    Scale(threshold_frame, from_=5.0, to=95.0, resolution=5.0, 
          orient="horizontal", variable=detection_threshold,
          length=200).pack(side=LEFT, padx=10)
    Label(threshold_frame, text="(值越小检测越敏感)").pack(side=LEFT)
    
    # 操作按钮
    Button(settings_frame, text="开始检测分割", 
           command=lambda: start_processing(True),  # 总是保留分割文件
           bg="green", fg="white", width=15, height=1).pack(pady=10)
    
    # 日志区域
    log_frame = LabelFrame(main_frame, text="处理日志", padx=10, pady=5)
    log_frame.pack(fill="both", expand=True)
    
    # 日志列表和滚动条
    log_list = Listbox(log_frame, height=15)
    scrollbar = Scrollbar(log_frame)
    log_list.pack(side=LEFT, fill="both", expand=True)
    scrollbar.pack(side=RIGHT, fill="y")
    log_list.config(yscrollcommand=scrollbar.set)
    scrollbar.config(command=log_list.yview)

def check_gpu():
    """检查 GPU 是否可用"""
    try:
        import cv2
        cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
        if cuda_count > 0:
            device = cv2.cuda.getDevice()
            props = cv2.cuda.getDeviceProperties(device)
            log(f"找到 NVIDIA GPU: {props.name()}")
            log(f"CUDA 计算能力: {props.major()}.{props.minor()}")
            log(f"总显存: {props.totalMemory() / 1024 / 1024:.1f} MB")
            return True
        else:
            log("未找到支持 CUDA 的 GPU，将使用 CPU 处理")
            return False
    except Exception as e:
        log(f"GPU 检测失败: {e}")
        return False

# 主程序入口
if __name__ == "__main__":
    # 初始化 Tkinter 界面
    root = Tk()
    root.title("一键视频混剪.QQ273356663")
    root.geometry("800x600")
    
    # 导入必要的 tkinter 组件
    from tkinter import Frame, LabelFrame, LEFT
    
    # 初始化全局变量
    init_globals(root)
    
    # 创建界面
    create_gui(root)
    
    # 设置图标
    try:
        if hasattr(sys, '_MEIPASS'):
            icon_path = os.path.join(sys._MEIPASS, "app.ico")
        else:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            icon_path = os.path.join(current_dir, "app.ico")
            
        if os.path.exists(icon_path):
            root.iconbitmap(icon_path)
            log("成功加载应用图标")
        else:
            log("未找到图标文件 app.ico")
    except Exception as e:
        log(f"加载图标失败: {e}")

    # 检查 FFmpeg
    if not check_ffmpeg():
        root.destroy()
        exit(1)
        
    # 检查 GPU
    check_gpu()
    
    # 启动主循环
    root.mainloop()
