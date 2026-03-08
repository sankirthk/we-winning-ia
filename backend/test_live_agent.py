import asyncio
import json

import websockets
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

JOB_ID = "test-live-job"
URL = f"ws://localhost:8080/api/live/{JOB_ID}"


async def run():
    try:
        async with websockets.connect(URL) as ws:
            print("Connected to live agent")

            await ws.send(
                json.dumps(
                    {
                        "type": "set_scene",
                        "scene_text": "Naive kernel fusion caused a 30x slowdown.",
                    }
                )
            )

            await ws.send(
                json.dumps(
                    {
                        "type": "text",
                        "text": "Why did kernel fusion slow down so much?",
                    }
                )
            )

            while True:
                message = await ws.recv()
                print("EVENT:", message)

    except ConnectionClosedOK as e:
        print(f"Socket closed normally: code={e.code}, reason={e.reason}")
    except ConnectionClosedError as e:
        print(f"Socket closed with error: code={e.code}, reason={e.reason}")


asyncio.run(run())