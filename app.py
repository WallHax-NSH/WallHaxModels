from fastapi import FastAPI, WebSocket
import os
import tempfile
from threedetr.infer_api import run_detection
import asyncio
import websockets
import json

app = FastAPI()

async def send_detection_json_to_producer(detection_json: dict):
    uri = "ws://192.168.5.232:8000/ws/producer/bounds"
    async with websockets.connect(uri) as websocket:
        await websocket.send(json.dumps(detection_json))

@app.websocket("/ws/consumer/plyStream")
async def ply_stream_consumer(websocket: WebSocket):
    uri = "ws://192.168.5.232:8000/ws/consumer/plyStream"
    async with websockets.connect(uri) as ws:
        ply_chunks = []
        tmp_path = None
        try:
            while True:
                chunk = await ws.recv()
                if chunk == "__END__":
                    break
                ply_chunks.append(chunk)
            # Reassemble file
            ply_data = "".join(ply_chunks)
            with tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8") as tmp:
                tmp.write(ply_data)
                tmp_path = tmp.name
            # Run detection
            result = run_detection(
                tmp_path,
                "threedetr/checkpoints/sunrgbd_masked_ep1080.pth",
                masked=True
            )
            os.unlink(tmp_path)
            # Send the full detection JSON to the producer WebSocket

            print(result)
            #await send_detection_json_to_producer(result)
            await ws.send(json.dumps(result))
        except Exception as e:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            await ws.send(json.dumps({"error": str(e)}))
        finally:
            await ws.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)