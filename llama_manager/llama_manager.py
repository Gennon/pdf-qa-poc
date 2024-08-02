import os
import platform
import shutil
import subprocess
import requests
import logging

class LlamaManager:
    def __init__(self, model_url, llamafile_version='0.8.11', download_dir='models', port=8080, embedding=False):
        self.model_url = model_url
        self.download_dir = download_dir
        self.model_path = os.path.join(download_dir, os.path.basename(model_url))
        self.pid = None
        self.port = port
        self.embedding = embedding
        self.lamafile_path = ''
        self.lamafile_version = llamafile_version
        self.setup_logging()
        self.download_llamafile()
        self.make_executable()
        

    def setup_logging(self):
        # Ensure the download directory exists
        os.makedirs(self.download_dir, exist_ok=True)
        self.log_file = os.path.join(self.download_dir, os.path.basename(self.model_url) + '.log')
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)


    def does_llamafile_exist(self):
        return os.path.exists(self.lamafile_path)
    
    def does_model_exist(self):
        return os.path.exists(self.model_path)

    def download_model(self):
        if self.does_model_exist():
            self.logger.info(f'Model already exists: {self.model_path}')
            return
        
        self.logger.info(f'Downloading model from: {self.model_url}')
        try:
          response = requests.get(self.model_url)
          response.raise_for_status()
        except Exception as e:
          self.logger.error(f'Error downloading model: {e}')
          return
        
        with open(self.model_path, 'wb') as file:
            file.write(response.content)
        self.logger.info(f'Downloaded model to {self.model_path}')


    def download_llamafile(self):
        self.lamafile_path = os.path.join(self.download_dir, f'llamafile-{self.lamafile_version}')
        if self.does_llamafile_exist():
            self.logger.info(f'Llamafile already exists: {self.lamafile_path}')
            return
        
        lamafile_url = f'https://github.com/Mozilla-Ocho/llamafile/releases/download/{self.lamafile_version}/llamafile-{self.lamafile_version}'
        try:
          response = requests.get(lamafile_url)
          response.raise_for_status()
        except Exception as e:
          self.logger.error(f'Error downloading llamafile: {e}')
          return
        
        with open(self.lamafile_path, 'wb') as file:
            file.write(response.content)
        self.logger.info(f'Downloaded llamafile to {self.lamafile_path}')


    def make_executable(self):
        # Check if file exists
        if not self.does_llamafile_exist():
            self.logger.error(f'Llamafile does not exist: {self.lamafile_path}')
            return
        
        if platform.system() == 'Windows':
            # Make a copy of the llamafile with a .exe extension
            shutil.copy(self.lamafile_path, self.lamafile_path + '.exe')
            self.lamafile_path += '.exe'           
        else:
            os.chmod(self.lamafile_path, 0o755)
        self.logger.info(f'Made llamafile executable: {self.lamafile_path}')


    def start_llamafile(self):
        # Check if files exists
        if not self.does_llamafile_exist():
            self.logger.error(f'Llamafile does not exist: {self.model_path}')
            return
        
        if not self.does_model_exist():
            self.logger.error(f'Model does not exist: {self.model_path}')
            return

        args = self.create_args()
        self.process = subprocess.Popen(
            [self.lamafile_path] + args, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            creationflags=subprocess.DETACHED_PROCESS)
        self.pid = self.process.pid
        self.create_pid_file()
        self.logger.info(f'Started llamafile with PID: {self.pid}')
        self.logger.info(f'Llamafile running on port: {self.port}')


    def create_args(self) -> list[str]:        
        args = ['-m', str(self.model_path),'--port', str(self.port), '--server', '--nobrowser']
        if self.embedding:
            args.append('--embedding')
        return args
    
    
    def check_health(self):
        if not os.path.exists(self.model_path):
            self.logger.error(f'File not found: {self.model_path}')
            return
        
        # Check if the llamafile is running
        try:
          response = requests.get(f'http://localhost:{self.port}/health')
          response.raise_for_status()
          status = response.json().get('status')
          self.logger.info(f'Health check status: {status}')
          return status
        except requests.exceptions.RequestException as e:
          self.logger.error(f'Error checking health: {e}')
          return 'error'
        

    def cleanup(self):
        self.logger.info('Cleaning up...')
        self.read_pid_file()
        import psutil
        if self.pid and psutil.pid_exists(self.pid):
            psutil.Process(self.pid).terminate()
            self.logger.info(f'Killed process with PID: {self.pid}')


    def create_pid_file(self):
        with open('llamafile.pid', 'w') as file:
            file.write(str(self.pid))


    def read_pid_file(self):
        if not os.path.exists('llamafile.pid'):
            return
        with open('llamafile.pid', 'r') as file:
            self.pid = int(file.read())


# Example usage:
# llama_manager = LlamaManager('http://example.com/llamafile')
# llama_manager.download_model()
# llama_manager.start_llamafile()
# llama_manager.cleanup()