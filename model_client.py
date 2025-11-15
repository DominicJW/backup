#accpets the arguments to pass to the model such as a tensor of input_ids, as well as other flags and stuff
#client: pickles input arguments
#client: sends them to server
#recieves pickeld outputs
#client: depickles outputs
#returns to user

import socket
import pickle
import os
from transformers import AutoConfig
from multiprocessing import shared_memory


class ModelClient:

    def __init__(self, socket_file='/tmp/model.sock'):
        self.socket_file = socket_file
        model_name = "mistralai/Mistral-7B-Instruct-v0.3"
        self.config = AutoConfig.from_pretrained(model_name)
    def __call__(self, **kwargs):
        """
        Send inputs to the model server and receive outputs.
        Usage: result = model_client(input_ids=tensor, attention_mask=tensor, ...)
        """
        # Create a socket connection
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            # Connect to the server
            sock.connect(self.socket_file)
            print("connected to server")
            

            # Pickle the inputs
            pickled_inputs = pickle.dumps(kwargs)#including pkv cache
            print(kwargs)
            data_size = len(pickled_inputs)
            input_shm = shared_memory.SharedMemory(create = True,size = data_size)
            input_shm.buf[:data_size] = pickled_inputs
            print("pickled dumped the inputs and written to shared memory")
            # Send the pickled data
            sock.sendall(input_shm.name.encode("utf-8"))
            sock.shutdown(socket.SHUT_WR)

            print("sent the input name")
            # Receive the pickled output
            output_shm_name = b''
            while True:
                chunk = sock.recv(8192)
                if not chunk:
                    break
                output_shm_name += chunk
            output_shm_name = output_shm_name.decode("utf-8")
            print("recieved the output name")
            print(output_shm_name)
            # Unpickle and return the output
            output_shm = shared_memory.SharedMemory(name = output_shm_name)
            output = pickle.loads(output_shm.buf[:])
            print("loaded from shared memory and depickled the output")
            output_shm.close()
            input_shm.close()
            return output
        
        finally:
            sock.close()

model = ModelClient()


#construct model inputs as kwargs #pickle the kwargs
#create inputs shared memory for the pickeld data 
#send inputs shmname
#recieve outputs shmname #read via pickle then close it



