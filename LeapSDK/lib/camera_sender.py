import websocket
import time

# Define the websocket server URL
websocket_url = "ws://localhost:8000"

# Define the data to send
data_to_send = "Hello, WebSocket!"

# Create a websocket connection
ws = websocket.create_connection(websocket_url)

while True:
    # Send the data
    ws.send(data_to_send)
    print("Data sent:", data_to_send)
    
    # Wait for some time before sending next data
    time.sleep(1)

    