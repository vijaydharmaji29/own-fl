import socket
from fl_main.lib.util.helpers import read_config, set_config_file
 
config_file = set_config_file("agent")
config = read_config(config_file)

HEADER = 64
PORT = config['comm_port']
SERVER = config['aggr_ip']
ADDR = (SERVER, PORT)
print("SERVER:", SERVER, "PORT:", PORT)
FORMAT = "utf-8"
DISCONNECT_MESSAGE = "!DISCONNECT"
DEREGISTER_MESSAGE = "!DEREGISTER"
NON_PARTICIPATION_MESSAGE = "!NONPARTICIPATE"

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

def send_non_participation_message():
    send(NON_PARTICIPATION_MESSAGE)

def send_deregister_message():
    send(DEREGISTER_MESSAGE)

