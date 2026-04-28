import argparse
import asyncio
import json
import urllib.parse
import urllib.request


def start_stream(base_url: str, stream_url: str, model: str, mode: str) -> dict:
	query = urllib.parse.urlencode({"url": stream_url, "model": model, "mode": mode})
	with urllib.request.urlopen(f"{base_url}/yolo/start?{query}", data=b"") as resp:
		return json.loads(resp.read().decode("utf-8"))


def stop_stream(base_url: str) -> dict:
	with urllib.request.urlopen(f"{base_url}/yolo/stop", data=b"") as resp:
		return json.loads(resp.read().decode("utf-8"))


async def recv_ws(ws_url: str, max_messages: int) -> None:
	try:
		import websockets
	except ImportError:
		print("缺少 websockets 包，请先安装: pip install websockets")
		return

	received = 0
	async with websockets.connect(ws_url, ping_interval=None) as ws:
		while received < max_messages:
			msg = await ws.recv()
			data = json.loads(msg)

			if data.get("type") == "heartbeat":
				print("[heartbeat]")
				continue

			mode = data.get("mode")
			count = data.get("count")
			if mode == "pose":
				people = data.get("people", [])
				kp_counts = [p.get("valid_kp_count", 0) for p in people]
				print(f"[pose] count={count}, valid_kp={kp_counts}")
			elif mode == "detect":
				boxes = data.get("boxes", [])
				print(f"[detect] count={count}, boxes={len(boxes)}")
			else:
				print(f"[unknown] {data}")

			received += 1


async def main() -> None:
	parser = argparse.ArgumentParser(description="YOLO realtime API smoke test")
	parser.add_argument("--base-url", default="http://127.0.0.1:8008", help="FastAPI base URL")
	parser.add_argument("--stream-url", required=True, help="RTSP/HLS/HTTP stream URL")
	parser.add_argument("--model", default="pose26l", help="model id/path")
	parser.add_argument("--mode", default="pose", choices=["auto", "detect", "pose"], help="inference mode")
	parser.add_argument("--messages", type=int, default=5, help="number of result messages to receive")
	args = parser.parse_args()

	ws_url = args.base_url.replace("http://", "ws://").replace("https://", "wss://") + "/yolo/ws/results"

	started = start_stream(args.base_url, args.stream_url, args.model, args.mode)
	print("start:", started)
	if started.get("status") not in {"started", "already_running"}:
		return

	try:
		await recv_ws(ws_url, args.messages)
	finally:
		stopped = stop_stream(args.base_url)
		print("stop:", stopped)


if __name__ == "__main__":
	asyncio.run(main())