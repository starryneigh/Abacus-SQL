import logging
import os
import configparser

class MyLogger:
    def __init__(self, name: str, log_file: str, level_config_file: str = 'config/logging_config.ini'):
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        self.logger = logging.getLogger(name)
        self._set_log_level_from_config(level_config_file)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)

        if not self.logger.handlers:
            self.logger.addHandler(file_handler)

    def _set_log_level_from_config(self, level_config_file: str):
        config = configparser.ConfigParser()

        # 检查配置文件是否存在
        if os.path.exists(level_config_file):
            config.read(level_config_file)

            # 检查配置文件是否有相应的设置
            if 'logging' in config and 'level' in config['logging']:
                level = config['logging']['level'].upper()

                # 使用配置中的日志等级
                self.logger.setLevel(getattr(logging, level, "INFO"))
            else:
                # 如果没有配置日志等级，则使用默认等级（例如 INFO）
                self.logger.setLevel(logging.INFO)
                print("No logging level set in config, defaulting to INFO.")
        else:
            # 如果配置文件不存在，设置为 INFO 模式
            self.logger.setLevel(logging.INFO)
            print("Config file not found, defaulting to INFO.")

    def set_level(self, level: int):
        """允许在运行时更改日志级别"""
        self.logger.setLevel(level)

    def debug(self, message: str):
        self.logger.debug(message)

    def info(self, message: str):
        self.logger.info(message)

    def warning(self, message: str):
        self.logger.warning(message)

    def error(self, message: str):
        self.logger.error(message)

    def critical(self, message: str):
        self.logger.critical(message)
