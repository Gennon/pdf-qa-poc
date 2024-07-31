import os
import platform
import subprocess
import requests
import logging

class LlamaManager:
    def __init__(self, llamafile_url, download_dir='/tmp', port=8080, embedding=False):
        self.llamafile_url = llamafile_url
        self.download_dir = download_dir
        self.llamafile_path = os.path.join(download_dir, os.path.basename(llamafile_url))
        self.pid = None
        self.port = port
        self.embedding = embedding
        self.setup_logging()

    def setup_logging(self):
        # Ensure the download directory exists
        os.makedirs(self.download_dir, exist_ok=True)
        self.log_file = os.path.join(self.download_dir, os.path.basename(self.llamafile_url) + '.log')
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
        return os.path.exists(self.llamafile_path)

    def download_llamafile(self):
        if self.does_llamafile_exist():
            self.logger.info(f'Llamafile already exists: {self.llamafile_path}')
            return
        
        self.logger.info(f'Downloading llamafile from: {self.llamafile_url}')
        try:
          response = requests.get(self.llamafile_url)
          response.raise_for_status()
        except Exception as e:
          self.logger.error(f'Error downloading llamafile: {e}')
          return
        
        with open(self.llamafile_path, 'wb') as file:
            file.write(response.content)
        self.logger.info(f'Downloaded llamafile to {self.llamafile_path}')

    def make_executable(self):
        # Check if file exists
        if not self.does_llamafile_exist():
            self.logger.error(f'Llamafile does not exist: {self.llamafile_path}')
            return
        
        if platform.system() == 'Windows':
            # Windows specific code to make file executable
            pass
        else:
            os.chmod(self.llamafile_path, 0o755)
        self.logger.info(f'Made llamafile executable: {self.llamafile_path}')

    def start_llamafile(self):
        # Check if file exists
        if not self.does_llamafile_exist():
            self.logger.error(f'Llamafile does not exist: {self.llamafile_path}')
            return
        
        args = self.create_args()
        process = subprocess.Popen([self.llamafile_path] + args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.pid = process.pid
        self.logger.info(f'Started llamafile with PID: {self.pid}')
        stdout, stderr = process.communicate()
        self.logger.info(f'Llamafile running on port: {self.port}')

    def create_args(self) -> list[str]:        
        args = ['--port', str(self.port), '--server', '--nobrowser']
        if self.embedding:
            args.append('--embedding')
        return args
    
    def check_health(self):
        if not os.path.exists(self.llamafile_path):
            self.logger.error(f'File not found: {self.llamafile_path}')
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
        if self.pid:
            os.kill(self.pid, 9)
            self.logger.info(f'Killed process with PID: {self.pid}')
        if os.path.exists(self.llamafile_path):
            os.remove(self.llamafile_path)
            self.logger.info(f'Removed llamafile: {self.llamafile_path}')
        if os.path.exists(self.log_file):
            os.remove(self.log_file)
            self.logger.info(f'Removed log file: {self.log_file}')

# Example usage:
# llama_manager = LlamaManager('http://example.com/llamafile')
# llama_manager.download_llamafile()
# llama_manager.make_executable()
# llama_manager.start_llamafile()
# llama_manager.cleanup()