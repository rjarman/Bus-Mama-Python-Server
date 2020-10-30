import threading
import socket
import os
import csv
from datetime import datetime
from libs.utils.config import Config

DEFAULT_DIRECTORY = os.path.dirname(__file__) + '/'

class ServerHandler(threading.Thread):
    def __init__(self):
        super().__init__()
        self.connections = []
    
    def run(self):

        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((context_manager.HOST, context_manager.PORT))

        server.listen(1)
        print('Listening at', server.getsockname())

        while True:
            connection_details, client_details = server.accept()
            print(f'Accepted a new connection from {connection_details.getpeername()} to {connection_details.getsockname()}')
            server_socket = ClientHandler(connection_details, client_details, self)
            server_socket.start()

            # Add thread to active connections
            self.connections.append(server_socket)
            print('Ready to receive messages from', connection_details.getpeername())

class ClientHandler(threading.Thread):
    def __init__(self, connection_details, client_details, main_server_ref):
        super().__init__()
        self.connection_details = connection_details
        self.__client_details = client_details
        self.__main_server_ref = main_server_ref
    
    def run(self):
        while True:
            client_message = self.connection_details.recv(1024).decode('ascii')
            server_message = '>>>>>' + ' '.join(client_message.split()[1:])
            server_message = server_message.encode('ascii')
            if client_message:
                print('{} says {!r}'.format(self.__client_details, client_message))
                # self.server.broadcast(message, self.sockname)
                self.connection_details.sendto(server_message, self.__client_details)
                context_manager.logger.log(self.__client_details, client_message, server_message)
            else:
                # Client has closed the socket, exit the thread
                print('{} has closed the connection'.format(self.__client_details))
                self.connection_details.close()
                self.__main_server_ref.remove_connection(self)
                return
                
class LogHandler:
    
    def __init__(self):
        self.__field_names = ['date_time', 'client_address', 'port', 'client_message', 'server_message']
        self.__log_file = open(context_manager.LOG_PATH, 'a', encoding='utf-8')
        self.__log_writer = csv.DictWriter(self.__log_file, fieldnames=self.__field_names)
        if not os.path.exists(context_manager.LOG_PATH): self.__log_writer.writeheader()

    def log(self, client_details, client_message, server_message):
        data_dict = {
            'date_time': datetime.now().isoformat(),
            'client_address': client_details[0],
            'port': client_details[1],
            'client_message': client_message,
            'server_message': server_message,
        }
        self.__log_writer.writerow(data_dict)

    def close(self):
        self.__log_file.close()

class ContextManager:
    def __init__(self):
        self.HOST, self.PORT = Config.SERVER['host'], Config.SERVER['port']
        self.LOG_PATH = Config.PATH['log']

        self.server = None

    def start_server(self):
        self.logger = LogHandler()
        self.init_threads()
    
    def init_threads(self):

        self.server = ServerHandler()
        self.server.start()

        terminate_server = threading.Thread(target = self.terminate_server, args = (self.server,))
        terminate_server.start()

    def terminate_server(self, server):
        while True:
            exit_command = input('')
            if exit_command == 'exit':
                print('Closing all connections...')
                for connection in server.connections:
                    connection.connection_details.close()
                print('Closing server_log file...')
                self.logger.close()
                print('Shutting down the server...')
                os._exit(0)

if __name__ == '__main__':
    context_manager = ContextManager()
    context_manager.start_server()