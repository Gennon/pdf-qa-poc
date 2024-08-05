from llama_manager import LlamaManager

llama_manager = LlamaManager('https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf')
llama_manager.download_model()
llama_manager.start_llamafile()

# # Check healt
llama_manager.check_health()

# Just to show that the cleanup works
#llama_manager.cleanup()
