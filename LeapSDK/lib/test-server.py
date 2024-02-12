import asyncio
import websockets
from collections import deque
import numpy as np

buffer_size = 30
circular_buffer = deque(maxlen=buffer_size)

def process_data(message):
    # Append data to the circular buffer
    circular_buffer.append(message)
    
    # Check if the buffer is full
    if len(circular_buffer) == buffer_size:
        # Convert the buffer to a numpy array
        buffer_data = np.array(list(circular_buffer))
        return buffer_data
    else:
        return None


async def handle_websocket(websocket, path):
    async for message in websocket:
        buffer_data = process_data(message)
        if buffer_data is not None:
            print(buffer_data)
        # print("Received message:", message)

start_server = websockets.serve(handle_websocket, 'localhost', 8000)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()

