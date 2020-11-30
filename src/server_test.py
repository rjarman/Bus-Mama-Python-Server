import asyncio
import datetime
import random
import websockets
import json
from libs import ContextManager
context_manager = ContextManager()

async def time(websocket, path):
    while True:
        recieved_message = await websocket.recv()
        recieved_message = json.loads(recieved_message)
        reply, tag = context_manager.make_reply(recieved_message)
        
        server_reply = json.dumps({'message': reply, 'tag': tag}, separators=(',',':'))
        await websocket.send(server_reply)
        await asyncio.sleep(random.random() * 3)

start_server = websockets.serve(time, "127.0.0.1", 8080)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()