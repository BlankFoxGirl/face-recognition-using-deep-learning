# This file is used to obtain frames from a socket and process them using the motion detection and facial recognition.
import socket, os, datetime, time, sys, json
activeStreams = []

# Methods


# Define the socket path
socket_path = 'processor.sock'

# Create a Unix socket
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

try:
    # Bind the socket to the specified path
    sock.bind(socket_path)

    # Listen for incoming connections
    sock.listen(1)
    print(f'Server listening on {socket_path}')

    while True:
        # Accept a client connection
        conn, addr = sock.accept()
        print(f'Client connected: {addr}')

        # Receive data from the client
        data = conn.recv(1024).decode('utf-8')
        print(f'Received data: {data}')

        # Parse the received JSON data
        try:
            json_data = json.loads(data)
            print('Parsed JSON data:')
            print(json_data)
        except json.JSONDecodeError as e:
            print(f'Error parsing JSON data: {e}')

        # Close the client connection
        conn.close()

except OSError as e:
    print(f'Error creating or binding socket: {e}')

finally:
    # Remove the socket file
    sock.close()
    print(f'Socket closed and file removed: {socket_path}')
    # os.remove(socket_path)
