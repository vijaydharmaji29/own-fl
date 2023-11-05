import socket

HEADER = 64
PORT = 5050
SERVER = "localhost"
ADDR = (SERVER, PORT)
FORMAT = "utf-8"
DISCONNECT_MESSAGE = "!DISCONNECT"
DEREGISTER_MESSAGE = "!DEREGISTER"

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(ADDR)

def send(msg):
    message = msg.encode(FORMAT)
    msg_length = len(message)
    send_length = str(msg_length).encode(FORMAT)
    send_length += b' ' * (HEADER - len(send_length))
    client.send(send_length)
    client.send(message)
    

def disconnect():
    send(DISCONNECT_MESSAGE)

def send_deregister_message():
    send(DEREGISTER_MESSAGE)

