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