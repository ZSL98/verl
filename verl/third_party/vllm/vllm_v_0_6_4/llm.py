import ray
from vllm import LLM

@ray.remote
class LLMRemote(LLM):
    def __init__(self):
        self.llm_instance = None

    def initialize_vllm(self, config):
        """Initialize the LLMEngine if not already initialized."""
        if self.llm_instance is None:
            print("Initializing the class instance...")
            self.llm_instance = LLM(model="deepseek-ai/deepseek-llm-7b-chat",
                                    tensor_parallel_size=8,  # Note this is a hack
                                    dtype=config.dtype,
                                    enforce_eager=config.enforce_eager,
                                    gpu_memory_utilization=config.gpu_memory_utilization,
                                    skip_tokenizer_init=False,
                                    max_model_len=config.prompt_length + config.response_length,
                                    load_format=config.load_format)
            print("Instance initialized.")
        else:
            print("Instance already initialized.")

    def call_instance_method(self, method_name, *args, **kwargs):
        """Call a method on the instance."""
        if self.llm_instance is None:
            raise ValueError("Instance has not been initialized yet.")
        method = getattr(self.llm_instance, method_name)
        return method(*args, **kwargs)