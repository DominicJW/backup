import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import socket
import pickle
import torch
import os
from multiprocessing import shared_memory
import sys


# Load the model once (you can move this to a separate module)
# model_name = "mistralai/Mistral-7B-v0.3"
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cuda",
    quantization_config=quant_config,
    dtype=torch.bfloat16
)
model.set_attn_implementation('eager')


def move_output_to_cpu(output):
    if isinstance(output, torch.Tensor):
        return output.cpu()
    elif hasattr(output, "_fields"):  # Handle named tuples
        return type(output)(**{k: move_output_to_cpu(getattr(output, k)) for k in output._fields})
    elif isinstance(output, (list, tuple)):
        return type(output)(move_output_to_cpu(x) for x in output)
    elif isinstance(output, dict):
        return {k: move_output_to_cpu(v) for k, v in output.items()}
    else:
        # If not a tensor or container, return as-is (no crash)
        return output




tokenizer = AutoTokenizer.from_pretrained(model_name)

# Server setup
socket_file = '/tmp/model.sock'

# Remove the socket file if it already exists
if os.path.exists(socket_file):
    os.remove(socket_file)

# Create the Unix socket
server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
server.bind(socket_file)
server.listen(1)
print("Model server is running...")

try:
    while True:
        conn, _ = server.accept()
        print("Received connection from client")

        # Receive data from the client
        inputs_shm_name = b''
        while True:
            chunk = conn.recv(8192)
            if not chunk:
                break
            inputs_shm_name += chunk
        inputs_shm_name = inputs_shm_name.decode("utf-8")
        inputs_shm = shared_memory.SharedMemory(name=inputs_shm_name)
        inputs = pickle.loads(inputs_shm.buf[:])
        inputs_shm.close()
        inputs["input_ids"] = inputs["input_ids"].to("cuda")
        try:
            inputs["past_key_value"] = inputs["past_key_value"].to("cuda")
        except:
            pass
        with torch.no_grad():
            outputs = model(**inputs)        
        outputs = move_output_to_cpu(outputs)
        pickled_output = pickle.dumps(outputs)
        data_size = len(pickled_output)
        output_shm = shared_memory.SharedMemory(create = True,size = data_size)
        output_shm.buf[:data_size] = pickled_output

        conn.sendall(output_shm.name.encode("utf-8"))
        conn.close()
        print("connection closed")
except KeyboardInterrupt:
    print("Shutting down model server...")
finally:
    server.close()
    if os.path.exists(socket_file):
        os.remove(socket_file)