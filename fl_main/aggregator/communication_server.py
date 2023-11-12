import socket
import threading
from .state_manager import StateManager
from fl_main.lib.util.helpers import read_config, set_config_file
 

config_file = set_config_file("aggregator")
config = read_config(config_file)

HEADER = 64
PORT = config['comm_port']
SERVER = config['aggr_ip']
print(SERVER)
print("SEVRER STARTED ON:", config['aggr_ip'], "PORT:", config['comm_port'])
ADDR = (SERVER, PORT)
FORMAT = "utf-8"
DISCONNECT_MESSAGE = "!DISCONNECT"
DEREGISTER_MESSAGE = "!DEREGISTER"



server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(ADDR)

def handle_client(conn, addr):
    print(f"[NEW CONNECTION] {addr} connected.")

    connected = True
    while connected:
        msg_length = conn.recv(HEADER).decode(FORMAT)
        
        if msg_length:
            msg_length = int(msg_length)
            msg = conn.recv(msg_length).decode(FORMAT)

            if msg == DISCONNECT_MESSAGE:
                connected = False

            if msg == DEREGISTER_MESSAGE:
                StateManager.DEREGISTERED += 1
                print("DEREGISTERED CLIENT")
                print("DERGISTER COUNT:", StateManager.DEREGISTERED)
                

            print(f"[{addr}] {msg}")

    conn.close()

def start_comm_server():
    server.listen()
    while True:
        conn, addr = server.accept()
        thread = threading.Thread(target=handle_client, args=(conn, addr))
        thread.start()
        print("[ACTIVE CONNECTIONS]:", threading.active_count() - 1)


if __name__ == "__main__":

    print("[STARTING] server is starting...")
    main_thread = threading.Thread(target=start_comm_server)
    main_thread.start()
    print("STARTED ABOVE THREAD")