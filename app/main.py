# app.py
import asyncio
import json
import os
import queue
import threading
import time

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.types import Scope

from .config import MODEL_REGISTRY, DEVICE, DEFAULT_IMGSZ, DEFAULT_CONF, MAX_STREAM_DURATION
from .inference import build_stream_generator, release_model, run_tiled_inference

app = FastAPI()

infer_semaphore = asyncio.Semaphore(1)  # 单图推理并发限制


class NoCacheStaticFiles(StaticFiles):
    """
    HLS文件(index.m3u8/.ts分片)默认会被浏览器缓存,导致内容更新后
    客户端还在读旧版本,表现为"画面对不上/播放的是很久以前的内容"。
    这里强制给所有响应加 no-cache,确保每次都拿到最新文件。
    """
    async def get_response(self, path: str, scope: Scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response


# HLS推流输出的根目录,通过静态文件服务对外暴露在 /hls 路径下。
# 因为全局只允许一路实时推理,固定用一个子目录"live"即可,
# 这样对外播放地址永远是同一个,不随model_name变化: /hls/live/index.m3u8
HLS_ROOT = "/data/hls_output"  # 按实际服务器可写路径调整
os.makedirs(os.path.join(HLS_ROOT, "live"), exist_ok=True)
app.mount("/hls", NoCacheStaticFiles(directory=HLS_ROOT), name="hls")

# 检测到目标时保存的截图,通过静态文件服务对外暴露在 /snapshots 路径下,
# 下游系统拿到SSE里的 "snapshot" 文件名后拼上这个前缀即可直接下载。
# 只是临时给下游一个下载窗口,不做长期持久化 —— 后台线程会定期清理
# 超过 SNAPSHOT_RETENTION_SECONDS 的旧文件,不需要手动维护磁盘空间。
SNAPSHOT_ROOT = "/data/snapshots"  # 按实际服务器可写路径调整
SNAPSHOT_RETENTION_SECONDS = 300   # 截图保留时长(秒),默认5分钟,按下游实际下载耗时调整
os.makedirs(SNAPSHOT_ROOT, exist_ok=True)
app.mount("/snapshots", StaticFiles(directory=SNAPSHOT_ROOT), name="snapshots")


def _snapshot_cleanup_loop():
    """
    后台常驻线程,每隔一段时间扫描一次SNAPSHOT_ROOT,删掉超过
    SNAPSHOT_RETENTION_SECONDS 的旧截图。用文件的实际修改时间判断
    年龄,不依赖文件名格式,更稳妥。
    """
    while True:
        now = time.time()
        try:
            for filename in os.listdir(SNAPSHOT_ROOT):
                filepath = os.path.join(SNAPSHOT_ROOT, filename)
                try:
                    if now - os.path.getmtime(filepath) > SNAPSHOT_RETENTION_SECONDS:
                        os.remove(filepath)
                except OSError:
                    pass  # 文件可能刚好被并发删除/正在写入,忽略即可
        except OSError:
            pass
        time.sleep(60)  # 每分钟扫描一次,不需要太频繁


@app.on_event("startup")
def _start_snapshot_cleanup():
    threading.Thread(target=_snapshot_cleanup_loop, daemon=True).start()


# ------------------ 单图推理接口 ------------------

@app.post("/custom/infer/{model_name}")
async def infer_image(
    model_name: str,
    file: UploadFile = File(...),
    conf: float = Form(DEFAULT_CONF),
    imgsz: int = Form(DEFAULT_IMGSZ),
    tiles: int = Form(1),  # 1=不切分, 4=2x2, 9=3x3
):
    if model_name not in MODEL_REGISTRY:
        return {"error": f"未知模型: {model_name}，可选: {list(MODEL_REGISTRY.keys())}"}
    if not (0.0 <= conf <= 1.0):
        return {"error": "conf 必须在 0~1 之间"}
    if tiles not in (1, 4, 9, 16):
        return {"error": "tiles 仅支持 1/4/9/16"}

    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "图片解码失败，请检查文件格式"}

    async with infer_semaphore:
        result = await run_in_threadpool(
            run_tiled_inference, model_name, MODEL_REGISTRY, img, conf, imgsz, DEVICE, tiles
        )

    return {"model": model_name, "conf": conf, "imgsz": imgsz, "tiles": tiles, **result}


# ------------------ 实时推理接口:全局只允许一路 ------------------

class StreamState:
    """用对象属性持有状态,闭包直接引用对象本身"""
    def __init__(self):
        self.current_token: str | None = None


stream_state = StreamState()
stream_lock = asyncio.Lock()


@app.get("/custom/infer_stream/{model_name}")
async def infer_stream(
    model_name: str,
    source: str,
    request: Request,
    conf: float = DEFAULT_CONF,
    imgsz: int = DEFAULT_IMGSZ,
    save_video_path: str | None = None,   # 传了就把标注后画面存成mp4,测试用
    save_video_fps: float = 5.0,           # 保存视频的播放fps,按实测吞吐大致填
    target_fps: float = 3.0,       # 固定采样密度抽帧,如 1.0 表示每秒推理1次
    push_hls: bool = False,                # True则同步把标注画面推成HLS,供展示系统播放
    push_hls_fps: float = 1.0,            # 推流输出帧率标注
    hls_segment_time: int = 2,             # 每个.ts分片时长(秒)
    hls_list_size: int = 5,                # index.m3u8保留的分片数量
    save_snapshot: bool = False,           # True则检测到目标时把这一帧直接落盘,供下游截图用
    probe_timeout: int = 15,               # ffprobe探测超时(秒),RTMP等协议可能需要调大
    enable_tracking: bool = False,         # True则给同一目标在连续帧间保持一致的track_id
    tracker: str = "bytetrack.yaml",       # 跟踪算法: bytetrack.yaml(快) 或 botsort.yaml(准)
    snapshot_new_targets_only: bool = False,  # True则只在出现新目标时才截图,需配合enable_tracking使用
    sse_new_targets_only: bool = False,    # True则只在出现新目标时才推送SSE,需配合enable_tracking使用
):
    """
    全局限制:同一时间只允许一路实时推理。
    新请求进来会自动顶替旧的,不需要前端做任何清理逻辑。

    实现说明: build_stream_generator 内部是 ffmpeg 拉流 + 逐帧同步推理,
    这部分重活放在独立的后台线程里跑,通过一个小容量的 queue.Queue
    把结果传回 async 主循环。这样"客户端断开/被顶替/超时"这三个检查
    可以持续以固定间隔轮询队列,不会被单帧推理的耗时卡住。

    target_fps: 不传则每一帧都推理;传入具体数值后按固定时间间隔抽帧,
    不管源视频真实帧率、不管推理速度如何,固定按这个密度采样。**如果
    同时开启了 enable_tracking,不建议再设置 target_fps** —— 跟踪依赖
    较高的帧间连续性才能正常关联同一目标,采样过稀疏会导致新目标长时间
    拿不到track_id。

    push_hls: 开启后,标注好检测框的画面会被持续编码并切片写入HLS格式,
    通过 /custom/hls/live/index.m3u8 对外提供播放地址,画面与检测框源自
    同一次处理,天然同步,解决展示系统与推理系统各自独立拉流导致的错位
    问题。展示系统应改为直接播放这个地址,而不是继续单独连原始视频源。

    save_snapshot: 开启后,每当这一帧检测到目标就直接落盘保存标注好的
    画面(jpg),用的是产生这条JSON结果的同一帧原始数据,不存在下游从
    视频画面里反推截图时刻的对齐误差。返回的JSON里会带 "snapshot" 字段
    (仅文件名),下游拼上 /custom/snapshots/{filename} 即可直接下载访问。

    enable_tracking: 开启后,同一个目标在连续帧之间会保持一致的track_id,
    而不是每帧独立重新分配ID,可用于去重计数、判断目标是否是新出现的。

    snapshot_new_targets_only: 需配合 enable_tracking=true。开启后只有
    出现本次会话里第一次见到的目标才截图,避免同一目标连续多帧被反复
    截图存成大量重复图片。

    sse_new_targets_only: 需配合 enable_tracking=true。开启后只有这一帧
    出现新目标时,这条JSON结果才会真正推送到SSE。推理本身仍然每帧都在
    跑(不受此参数影响,只影响对外输出频率),推荐搭配"不设置target_fps"
    一起使用,让跟踪拿到完整的帧序列以保证关联准确,只是减少下游需要
    处理的消息数量 —— 没有新目标的帧不再占用SSE通道。
    """
    if model_name not in MODEL_REGISTRY:
        return {"error": f"未知模型: {model_name}，可选: {list(MODEL_REGISTRY.keys())}"}
    if not (0.0 <= conf <= 1.0):
        return {"error": "conf 必须在 0~1 之间"}
    if snapshot_new_targets_only and not enable_tracking:
        return {"error": "snapshot_new_targets_only 需要配合 enable_tracking=true 使用,否则无法判断目标是否为新出现"}
    if sse_new_targets_only and not enable_tracking:
        return {"error": "sse_new_targets_only 需要配合 enable_tracking=true 使用,否则无法判断目标是否为新出现"}

    my_token = f"{model_name}:{time.time_ns()}"
    async with stream_lock:
        stream_state.current_token = my_token

    push_hls_dir = os.path.join(HLS_ROOT, "live") if push_hls else None
    snapshot_dir = SNAPSHOT_ROOT if save_snapshot else None

    async def event_generator():
        config = MODEL_REGISTRY[model_name]
        model = config["loader"](config["weights"])
        start_time = time.time()

        stop_event = threading.Event()
        # maxsize 故意设小: 如果消费跟不上(比如网络推送慢),
        # 宁可让新结果覆盖式丢弃旧结果,也不要让队列无限堆积增加延迟
        result_queue: queue.Queue = queue.Queue(maxsize=2)

        def worker():
            try:
                for parsed in build_stream_generator(
                    model, source, conf, imgsz, DEVICE,
                    stop_event=stop_event,
                    save_video_path=save_video_path,
                    save_video_fps=save_video_fps,
                    target_fps=target_fps,
                    push_hls_dir=push_hls_dir,
                    push_hls_fps=push_hls_fps,
                    hls_segment_time=hls_segment_time,
                    hls_list_size=hls_list_size,
                    snapshot_dir=snapshot_dir,
                    probe_timeout=probe_timeout,
                    enable_tracking=enable_tracking,
                    tracker=tracker,
                    snapshot_new_targets_only=snapshot_new_targets_only,
                    sse_new_targets_only=sse_new_targets_only,
                ):
                    if stop_event.is_set():
                        break
                    try:
                        result_queue.put(parsed, timeout=1)
                    except queue.Full:
                        pass  # 消费跟不上,丢弃这帧,不阻塞ffmpeg读取循环
            except Exception as e:
                try:
                    result_queue.put({"error": f"推理线程异常: {e}"}, timeout=1)
                except queue.Full:
                    pass
            finally:
                try:
                    result_queue.put(None, timeout=1)  # 哨兵值: 线程已结束
                except queue.Full:
                    pass

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        loop = asyncio.get_event_loop()

        try:
            while True:
                if await request.is_disconnected():
                    break
                if stream_state.current_token != my_token:
                    yield f"data: {json.dumps({'error': '已被新的推理请求顶替,自动停止'})}\n\n"
                    break
                if time.time() - start_time > MAX_STREAM_DURATION:
                    yield f"data: {json.dumps({'error': '已达最大运行时长,自动停止'})}\n\n"
                    break

                try:
                    # 短超时轮询队列,既不空转浪费CPU,也能及时回到上面三个检查
                    item = await loop.run_in_executor(None, result_queue.get, True, 1.0)
                except queue.Empty:
                    continue

                if item is None:
                    break  # worker线程正常结束(流中断/被停止)

                yield f"data: {json.dumps(item)}\n\n"

                if "error" in item:
                    break

                await asyncio.sleep(0)

        finally:
            stop_event.set()
            thread.join(timeout=5)
            release_model(model)
            async with stream_lock:
                if stream_state.current_token == my_token:
                    stream_state.current_token = None

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/health")
async def health():
    return {"status": "ok", "models_available": list(MODEL_REGISTRY.keys())}