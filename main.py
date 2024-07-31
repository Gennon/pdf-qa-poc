from llama_manager import LlamaManager

llama_manager = LlamaManager('https://huggingface.co/Mozilla/TinyLlama-1.1B-Chat-v1.0-llamafile/blob/main/TinyLlama-1.1B-Chat-v1.0.Q5_K_M.llamafile')
llama_manager.download_llamafile()
llama_manager.make_executable()
llama_manager.start_llamafile()

# # Check health
print(llama_manager.check_health())
# llama_manager.cleanup()
