# inference.py
import gc
import subprocess
import threading
import time as _time

import numpy as np
import torch
from .tiling import split_image, merge_detections, offset_detection


def parse_detection(result) -> list[dict]:
    boxes = result.boxes
    if boxes is None:
        return []
    dets = []
    for b in boxes:
        cls_id = int(b.cls)
        d = {
            "cls": cls_id,
            "cls_name": result.names[cls_id],  # 类别名称,来自模型自带的names映射
            "conf": float(b.conf),
            "xyxy": b.xyxy[0].tolist(),
        }
        # track_id 只有在用 model.track() 且这一帧成功跟踪到该目标时才存在,
        # 单图推理(model.predict())或跟踪未确认时,这个key直接不出现,
        # 而不是以 null 的形式占位。
        if b.id is not None:
            d["track_id"] = int(b.id[0])
        dets.append(d)
    return dets


def parse_segmentation(result) -> list[dict]:
    masks = result.masks
    boxes = result.boxes
    if masks is None or boxes is None:
        return []
    dets = []
    for b, m in zip(boxes, masks):
        cls_id = int(b.cls)
        d = {
            "cls": cls_id,
            "cls_name": result.names[cls_id],
            "conf": float(b.conf),
            "xyxy": b.xyxy[0].tolist(),
            "polygon": m.xy[0].tolist(),
        }
        if b.id is not None:
            d["track_id"] = int(b.id[0])
        dets.append(d)
    return dets


def parse_pose(result) -> list[dict]:
    keypoints = result.keypoints
    boxes = result.boxes
    if keypoints is None or boxes is None:
        return []
    dets = []
    for b, kp in zip(boxes, keypoints):
        cls_id = int(b.cls)
        d = {
            "cls": cls_id,
            "cls_name": result.names[cls_id],
            "conf": float(b.conf),
            "xyxy": b.xyxy[0].tolist(),
            "keypoints": kp.xy[0].tolist(),
            "kp_conf": kp.conf[0].tolist() if kp.conf is not None else None,
        }
        if b.id is not None:
            d["track_id"] = int(b.id[0])
        dets.append(d)
    return dets


def parse_result(result) -> dict:
    if result.masks is not None:
        return {"task": "segmentation", "detections": parse_segmentation(result)}
    if result.keypoints is not None:
        return {"task": "pose", "detections": parse_pose(result)}
    if result.boxes is not None:
        return {"task": "detection", "detections": parse_detection(result)}
    return {"task": "unknown", "detections": []}


def release_model(model) -> None:
    del model
    gc.collect()
    torch.cuda.empty_cache()


def run_single_inference(
    model_name: str,
    model_registry: dict,
    img,
    conf: float,
    imgsz: int,
    device: int,
) -> dict:
    config = model_registry[model_name]
    model = config["loader"](config["weights"])
    try:
        results = model.predict(
            img, imgsz=imgsz, conf=conf,
            device=device, verbose=False, half=True,
        )
        return parse_result(results[0])
    finally:
        release_model(model)


# ------------------ 流式推理: ffmpeg 低延迟拉流 ------------------

def probe_resolution(url: str, timeout: int = 15) -> tuple[int, int]:
    """
    用ffprobe探测视频源宽高。ffmpeg以rawvideo管道输出裸像素数据时,
    没有容器封装信息,必须提前知道宽高才能正确 reshape 出图像。

    注意: 如果url是HLS的master playlist(多码率自适应索引),ffprobe会
    把每一档分辨率都列出来(多行输出),这里只取第一行,不假设只有一路流。

    timeout: RTMP等协议的探测耗时可能明显长于HLS,固定的短超时可能不够用,
    必要时可由调用方传入更大的值。
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=s=x:p=0",
        url,
    ]
    out = subprocess.check_output(cmd, timeout=timeout).decode().strip()
    first_line = out.splitlines()[0]
    w_str, h_str = first_line.split("x")
    return int(w_str), int(h_str)


def build_stream_generator(
    model, source: str, conf: float, imgsz: int, device: int,
    stop_event: threading.Event | None = None,
    save_video_path: str | None = None,
    save_video_fps: float = 5.0,
    target_fps: float | None = None,
    push_hls_dir: str | None = None,
    push_hls_fps: float = 15.0,
    hls_segment_time: int = 3,
    hls_list_size: int = 5,
    snapshot_dir: str | None = None,
    probe_timeout: int = 15,
    enable_tracking: bool = False,
    tracker: str = "bytetrack.yaml",
    snapshot_new_targets_only: bool = False,
    sse_new_targets_only: bool = False,
):
    """
    用ffmpeg子进程以低延迟参数拉流(-fflags nobuffer -flags low_delay),
    逐帧读取原始像素数据后手动喂给模型做单帧推理。

    相比直接把 source 交给 model.predict(source=..., stream=True):
    Ultralytics 内部对 HLS 这类网络流的探测/缓冲逻辑会带来很高的首帧延迟
    (实测过57秒),用ffmpeg专门的低延迟参数拉流能避开这个问题。

    stop_event: 由调用方传入,用于从外部提前中止这个生成器(比如客户端断开/
    被新请求顶替时),否则生成器会一直阻塞在 proc.stdout.read() 上等新数据。

    save_video_path: 传入文件路径(如 "/path/to/output.mp4")则把带检测框标注
                    的每一帧写成视频文件,不传则不保存。测试/调试时用,持续
                    跑的生产流式服务不建议开,视频会一直增长占用磁盘。
    save_video_fps: 输出视频的fps。因为是逐帧手动写入,这个值只影响播放速度
                    (视频里存了多少帧就是多少帧,不会自动补帧),按你实际推理
                    吞吐大致估一个就行,不需要跟源视频真实fps一致。

    target_fps: 固定采样密度抽帧。不传(None)则每一帧都推理(原行为)。
                传入具体数值(如 1.0)后,不管源视频真实帧率是多少、不管
                推理速度如何,固定按"每隔 1/target_fps 秒推理一次"的
                节奏采样。期间读到的帧仍会正常从管道读出(避免ffmpeg写端
                被阻塞导致源头卡顿),只是直接丢弃不做推理,不送进模型。

    push_hls_dir: 传入一个目录路径,则把带检测框标注的画面持续编码并切片
                写成HLS格式(index.m3u8 + 若干.ts分片),输出结构跟大疆那种
                HLS源一致,可以直接配合静态文件服务对外提供播放地址,
                彻底解决"展示系统和推理系统各自独立拉流导致画面与检测框
                对不齐"的问题 —— 展示系统改为直接播放这一路,不再单独去
                连原始源。不传则不做这路推流(原行为)。
    push_hls_fps: 推流输出的帧率标注,**必须与实际写入帧的真实节奏一致**
                (即 target_fps,若未设置target_fps则应对应源视频真实帧率),
                否则会导致编码时间轴与真实到达速度不匹配,播放内容逐渐
                落后于当前时间。
    hls_segment_time: 每个.ts分片的时长(秒)。越短延迟越低,但请求频率
                和文件数量越多,常见取值1-4秒。
    hls_list_size: index.m3u8中保留的分片数量,超出的旧分片会被自动删除,
                避免磁盘无限增长。

    snapshot_dir: 传入一个目录路径后,每当这一帧检测到目标(detections非空)
                就把这一帧的原始画面(不带检测框/分割/关键点标注)直接落盘
                保存为jpg,文件名是毫秒级时间戳。这是给下游系统截图用的
                最准确方式 —— 用的是产生这条JSON结果的同一帧原始数据,
                不存在"从视频画面里反推该在哪个时刻截图"的对齐误差。
                返回的JSON里会带上 "snapshot" 字段(仅文件名),下游拼上
                静态文件服务的URL前缀即可直接下载访问。

    probe_timeout: ffprobe探测视频源宽高的超时时间(秒)。HLS通常很快,
                RTMP等协议可能耗时更长,超时会直接判为失败并抛出异常。

    enable_tracking: True时用 model.track() 代替 model.predict(),
                给同一个目标在连续帧之间保持一致的track_id,而不是每帧
                独立重新分配ID。要求同一个model实例持续处理整个会话
                (本函数天然满足这一点,model只在会话开始时加载一次)。
                每条detection结果会带上 "track_id" 字段,跟踪暂时丢失
                该目标时为 null。切片推理(run_tiled_inference)不支持
                跟踪,因为各tile是独立区域,不存在有意义的帧间连续性。
    tracker: 跟踪算法配置文件,"bytetrack.yaml"(默认,速度快,基于位置
                关联)或 "botsort.yaml"(精度更高,额外用外观特征辅助
                关联,速度稍慢)。

    snapshot_new_targets_only: 需配合 enable_tracking=True 使用。True时
                只有出现"本次会话里第一次见到的track_id"才截图,避免同一
                目标连续多帧被反复截图;False(默认)时只要这一帧有检测
                目标就截图,是原来的行为。截图触发时JSON里会额外带上
                "new_track_ids" 字段,列出这次是哪些新目标触发的截图。

    sse_new_targets_only: 需配合 enable_tracking=True 使用。True时,只有
                这一帧出现新目标才会真正 yield 这条JSON结果(即只有新
                目标才会被推送到SSE)。推理本身仍然每帧都在跑,不受影响
                (跟踪需要较高的帧间连续性才能正常关联,不建议同时配合
                target_fps 限制推理频率,否则新目标可能长时间拿不到
                track_id,导致这个过滤条件迟迟不满足)。video/推流的写入
                不受此参数影响,仍然每帧照常写入,只影响SSE输出频率。
    """
    W, H = probe_resolution(source, timeout=probe_timeout)
    frame_size = W * H * 3  # bgr24, 每像素3字节

    if snapshot_dir:
        import os
        os.makedirs(snapshot_dir, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-fflags", "nobuffer",
        "-flags", "low_delay",
        "-i", source,
        "-pix_fmt", "bgr24",
        "-vcodec", "rawvideo",
        "-an",
        "-f", "rawvideo",
        "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    if proc.stdout is None:
        raise RuntimeError("ffmpeg stdout pipe 不可用")

    video_writer = None
    if save_video_path:
        import cv2
        import os
        os.makedirs(os.path.dirname(save_video_path) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
        video_writer = cv2.VideoWriter(save_video_path, fourcc, save_video_fps, (W, H))

    # snapshot功能独立于save_video_path,单独确保cv2已导入
    if snapshot_dir:
        import cv2

    push_proc = None
    if push_hls_dir:
        import os
        import glob
        os.makedirs(push_hls_dir, exist_ok=True)
        # 每次新开一路推流前先清空旧文件,避免残留的旧分片/索引被误读到
        # (比如上一次测试用不同fps/参数生成的过期内容,新一轮开始前必须清干净)
        for old_file in glob.glob(os.path.join(push_hls_dir, "*")):
            try:
                os.remove(old_file)
            except OSError:
                pass

        # 关键帧间隔必须显式对齐分片时长,否则libx264默认GOP过大,
        # HLS切片只能等到关键帧才能切,导致分片时长远超hls_segment_time
        # (实测出现过87秒/165秒的异常分片),且迟迟凑不够hls_list_size
        # 形成正常的滑动窗口,表现为播放器把它当成一段"已结束的短视频"
        # 而不是持续滚动的直播(还会带上不该出现的 #EXT-X-ENDLIST 标记)。
        keyframe_interval = max(1, int(hls_segment_time * push_hls_fps))

        push_cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{W}x{H}",
            "-r", str(push_hls_fps),
            "-i", "pipe:0",           # 从标准输入读取标注后的帧
            "-c:v", "libx264",
            "-preset", "ultrafast",   # 优先低延迟,不追求压缩率/画质
            "-tune", "zerolatency",
            "-g", str(keyframe_interval),          # 强制关键帧间隔=分片时长
            "-keyint_min", str(keyframe_interval),
            "-sc_threshold", "0",     # 关闭场景切换自适应关键帧,保证间隔精确
            # 显式限制码率,避免x264按画面内容复杂度自由发挥导致带宽尖峰。
            # push_hls_fps很低时,相邻"帧"对应完整1秒的真实画面变化(不像
            # 正常30fps里相邻帧几乎一样),P帧体积可能不小,不加码率上限
            # 容易在画面复杂/变化剧烈时产生瞬时大数据量,超出播放端实际
            # 带宽就会造成缓冲卡顿。
            "-b:v", "800k",
            "-maxrate", "1000k",
            "-bufsize", "2000k",
            "-pix_fmt", "yuv420p",    # 保证播放器兼容性
            "-f", "hls",
            "-hls_time", str(hls_segment_time),
            "-hls_list_size", str(hls_list_size),
            "-hls_flags", "delete_segments+append_list",
            "-hls_segment_filename", os.path.join(push_hls_dir, "seg_%05d.ts"),
            os.path.join(push_hls_dir, "index.m3u8"),
        ]
        push_proc = subprocess.Popen(
            push_cmd, stdin=subprocess.PIPE,
            stderr=open("/tmp/ffmpeg_push_stderr.log", "wb"),  # 临时: 捕获推流ffmpeg的报错日志,便于排查崩溃原因
        )

    sample_interval = (1.0 / target_fps) if target_fps else 0.0
    last_infer_time = 0.0

    # 记录本次会话里已经出现过的track_id,用于判断"这个目标是不是新出现的"。
    # 只在会话生命周期内有效,新开一次流式会话(重新加载模型)会重新清空,
    # 符合"每次巡检独立计数"的直觉。
    seen_track_ids: set[int] = set()

    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                break

            raw = proc.stdout.read(frame_size)
            if len(raw) < frame_size:
                break  # 流中断或结束

            # 固定采样密度模式: 没到目标间隔的帧直接丢弃,不占推理资源,
            # 但必须先读出来(上面这行已经读了),不能跳过读取本身,
            # 否则ffmpeg管道写端会被阻塞,反过来拖慢/卡住拉流本身。
            if target_fps:
                now = _time.time()
                if now - last_infer_time < sample_interval:
                    continue
                last_infer_time = now

            # 在读到这一帧的时刻打时间戳,而不是等推理跑完再打 ——
            # 推理本身有耗时,用"读帧时刻"更准确反映这一帧对应的真实时间点,
            # 方便前端/下游用来跟展示画面的时间做粗略比对,判断当前的滞后程度。
            frame_time = _time.time()

            frame = np.frombuffer(raw, dtype=np.uint8).reshape((H, W, 3))

            if enable_tracking:
                results = model.track(
                    source=frame, imgsz=imgsz, conf=conf,
                    device=device, half=True, verbose=False,
                    persist=True,      # 保持跟踪状态,不是每帧重新分配ID
                    tracker=tracker,
                )
            else:
                results = model.predict(
                    source=frame, imgsz=imgsz, conf=conf,
                    device=device, half=True, verbose=False,
                )

            # snapshot现在存的是原始画面(不带标注),不再需要annotated,
            # 所以这里不把 snapshot_dir 计入 need_annotated 的判断条件。
            need_annotated = video_writer is not None or push_proc is not None
            annotated = results[0].plot() if need_annotated else None

            if video_writer is not None and annotated is not None:
                video_writer.write(annotated)

            if push_proc is not None and push_proc.stdin is not None and annotated is not None:
                try:
                    push_proc.stdin.write(annotated.tobytes())
                except BrokenPipeError:
                    # 推流进程意外退出(比如磁盘写满/权限问题),
                    # 不让这个异常打断主推理循环,但停止继续尝试推流。
                    push_proc = None

            parsed = parse_result(results[0])
            parsed["timestamp"] = frame_time  # unix时间戳(秒,含小数),UTC

            # 检测到目标时直接落盘这一帧,而不是让下游从视频画面里反推
            # "该在哪个时刻截图" —— 这里用的就是产生这条JSON结果的同一帧
            # 原始数据,零延迟、零对齐误差,是最准确的截图方式。
            #
            # snapshot_new_targets_only=True 时,只有出现"本次会话里第一次
            # 见到的track_id"才截图,避免同一个目标连续多帧被反复截图存成
            # 大量几乎一样的图片。依赖 enable_tracking=True 提供的track_id
            # 才能判断"新旧",没开跟踪时这个开关不生效,退化为原来的行为。
            current_ids = {
                d["track_id"] for d in parsed["detections"] if "track_id" in d
            }
            new_ids = current_ids - seen_track_ids
            seen_track_ids |= current_ids

            has_detections = len(parsed["detections"]) > 0
            should_snapshot = has_detections and (
                not snapshot_new_targets_only or len(new_ids) > 0
            )

            if snapshot_dir is not None and should_snapshot:
                import os
                filename = f"{int(frame_time * 1000)}.jpg"
                filepath = os.path.join(snapshot_dir, filename)
                cv2.imwrite(filepath, frame)  # 存原始画面,不带检测框/分割/关键点标注
                parsed["snapshot"] = filename  # 下游拼上静态文件服务前缀即可访问
                if snapshot_new_targets_only:
                    parsed["new_track_ids"] = sorted(new_ids)  # 告诉下游这次是哪些新目标触发的

            # sse_new_targets_only=True 时,只有这一帧出现新目标才真正往外
            # 推送这条JSON结果 —— 推理本身仍然每帧都在跑(保证跟踪关联的
            # 准确性,跟踪需要较高的帧间连续性才能正常工作),只是把"处理
            # 频率"和"对外输出频率"解耦开:没有新目标的帧,视频/推流该写
            # 的还是照常写,只是不占用SSE这条通道,减少下游要处理的消息量。
            should_yield = (not sse_new_targets_only) or (len(new_ids) > 0)
            if should_yield:
                yield parsed
    finally:
        proc.kill()
        proc.wait()
        if video_writer is not None:
            video_writer.release()
        if push_proc is not None:
            if push_proc.stdin is not None:
                push_proc.stdin.close()
            push_proc.wait(timeout=10)


# ------------------ 切片推理(单图) ------------------

def run_tiled_inference(model_name, registry, img, conf, imgsz, device, grid: int = 1):
    """
    grid=1 时退化为普通单图推理,grid=4/9 时做切片推理。
    模型只加载一次,所有tile一次性组batch推理,xyxy/polygon/keypoints
    统一走 offset_detection 做坐标偏移映射回原图坐标系。
    """
    if grid == 1:
        return run_single_inference(model_name, registry, img, conf, imgsz, device)

    config = registry[model_name]
    model = config["loader"](config["weights"])

    try:
        tiles = split_image(img, grid)
        tile_imgs = [t[0] for t in tiles]
        offsets = [(t[1], t[2]) for t in tiles]

        results = model.predict(
            tile_imgs, imgsz=imgsz, conf=conf,
            device=device, verbose=False, half=True,
        )

        task = "unknown"
        all_dets = []
        for result, (x_off, y_off) in zip(results, offsets):
            parsed = parse_result(result)
            task = parsed["task"]
            for det in parsed["detections"]:
                all_dets.append(offset_detection(det, x_off, y_off))

        merged = merge_detections(all_dets)
        return {"task": task, "detections": merged, "count": len(merged)}
    finally:
        release_model(model)