import os
import torch
from typing import Optional

import deepspeed
import logging
logging.basicConfig(format='[%(asctime)s] %(filename)s %(funcName)s():%(lineno)i [%(levelname)s] %(message)s', level=logging.DEBUG)
from djl_python.inputs import Input
from djl_python.outputs import Output
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

predictor = None


def get_model(properties: dict):
    model_dir = properties.get("model_dir")
    model_id = properties.get("model_id")
    mp_size = int(properties.get("tensor_parallel_degree", "2"))
    local_rank = int(os.getenv('OMPI_COMM_WORLD_LOCAL_RANK', '0'))
    logging.info(f"process [{os.getpid()}  rank is [{local_rank}]]")
    if not model_id:
        model_id = model_dir
    logging.info(f"rank[{local_rank}] start load model")
    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    logging.info(f"rank[{local_rank}] success load model")

    model = deepspeed.init_inference(model,
                                     mp_size=mp_size,
                                     dtype=torch.float16,
                                     replace_method='auto',
                                     replace_with_kernel_inject=True)
    logging.info(f"rank[{local_rank}] success to convert model to deepspeed kernel")

    return pipeline(task='text-generation',
                    model=model,
                    tokenizer=tokenizer,
                    device=local_rank)


def handle(inputs: Input) -> Optional[Output]:
    global predictor
    if not predictor:
        predictor = get_model(inputs.get_properties())

    if inputs.is_empty():
        # Model server makes an empty call to warmup the model on startup
        return None

    data = inputs.get_as_string()
    output = Output()
    output.add_property("content-type", "application/json")
    result = predictor(data, do_sample=True, max_new_tokens=50)
    return output.add(result)


