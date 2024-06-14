import sys
# import gradio as gr
import torch
torch.set_num_threads(1)
import transformers
import argparse
from tqdm import tqdm
import json
import tarfile
import os
import wandb
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig
from transformers import AutoModelForCausalLM
from sklearn.metrics import roc_auc_score
from importlib.metadata import version
import subprocess
from sklearn.metrics import roc_auc_score

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def run_command(command):
    """Run a system command and return the output."""
    try:
        output = subprocess.check_output(command, stderr=subprocess.STDOUT).decode("utf-8")
        return True, output
    except subprocess.CalledProcessError as e:
        return False, e.output.decode("utf-8")


run_command(["pip", "install", "bitsandbytes-cuda116"])


def check_cuda_availability():
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch built with CUDA: {torch.version.cuda}")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA is available to PyTorch: {cuda_available}")
    if cuda_available:
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
    
    success, _ = run_command(["python", "-c", "import torch; torch.tensor([1.0], device='cuda:0')"])
    if success:
        print("Successfully created a tensor on CUDA.")
    else:
        print("Failed to create a tensor on CUDA. This might be due to no CUDA devices being available.")


def check_env_variables():
    cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", "Not set")
    print(f"CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")
    
    ld_library_path = os.getenv("LD_LIBRARY_PATH", "Not set")
    print(f"LD_LIBRARY_PATH: {ld_library_path}")


def list_and_print_contents(dir_path):
    print(f"Contents of {dir_path}:")
    try:
        # List all files and directories in the specified path
        entries = os.listdir(dir_path)
        if entries:
            for entry in entries:
                print(entry)
        else:
            print("The directory is empty.")
    except FileNotFoundError:
        print("The directory does not exist.")


def start_evaluate(args):

    # Print out the arguments
    print(
        f"Evaluating Alpaca-LoRA model with params:\n"
        f"load_8bit: {args.load_8bit}\n"
        f"base_model: {args.base_model}\n"
        f"lora_weights: {args.lora_weights}\n"
        f"test_data_path: {args.test_data_path}\n"
        f"result_json_data: {args.result_json_data}\n"
        f"batch_size: {args.batch_size}\n"
        f"share_gradio: {args.share_gradio}\n"
    )
    # Assign arguments to variables
    load_8bit = args.load_8bit
    base_model = args.base_model
    lora_weights = args.lora_weights
    test_data_path = args.test_data_path
    result_json_data = args.result_json_data
    batch_size = args.batch_size
    share_gradio = args.share_gradio


    # base_model = base_model + "/adapter_model.bin"
    model_type = lora_weights.split('/')[-1]
    model_name = '_'.join(model_type.split('_')[:2])

    if model_type.find('book') > -1:
        train_sce = 'book'
    else:
        train_sce = 'movie'
    
    if test_data_path.find('book') > -1:
        test_sce = 'book'
    else:
        test_sce = 'movie'
    
    print(f"Model type: {model_type}")
    print(f"Model name: {model_name}")
    
    # temp_list = model_type.split('_')
    # print(f"temp_list: {temp_list}")
    # seed = temp_list[-2]
    # sample = temp_list[-1]
    
    if os.path.exists(result_json_data):
        f = open(result_json_data, 'r')
        data = json.load(f)
        f.close()
    else:
        data = dict()

    if not data.__contains__(train_sce):
        data[train_sce] = {}
    if not data[train_sce].__contains__(test_sce):
        data[train_sce][test_sce] = {}
    if not data[train_sce][test_sce].__contains__(model_name):
        data[train_sce][test_sce][model_name] = {}

    # if not data[train_sce][test_sce][model_name].__contains__(seed):
    #     data[train_sce][test_sce][model_name][seed] = {}
    # if data[train_sce][test_sce][model_name][seed].__contains__(sample):
    #     exit(0)
        # data[train_sce][test_sce][model_name][seed][sample] = 


    # quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)
    tokenizer = LlamaTokenizer.from_pretrained("baffo32/decapoda-research-llama-7B-hf", model_max_length=512)
    # device = "cpu"

    print("Loaded the Tokenizer and now loading the model")
    if device == "cuda":
        print("Using CUDA")
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
            # quantization_config=quantization_config
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
            device_map="auto",
            # device_map={'': 0}
        )
    elif device == "mps":
        print("Using mps")
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        print("Using cpus")
        model = LlamaForCausalLM.from_pretrained(
            "baffo32/decapoda-research-llama-7B-hf", device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )
    
    # Enable model parallelism
    model.is_parallelizable = True
    model.model_parallel = True

    tokenizer.padding_side = "left"
    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        print("Loading model in half")
        # model.half()  # seems to fix bugs for some users.

    print("Compiling model")
    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    
    def evaluate(
        instructions,
        inputs=None,
        temperature=0,
        top_p=1.0,
        top_k=40,
        num_beams=1,
        max_new_tokens=128,
        **kwargs,
    ):
        print("generating prompt")
        prompt = [generate_prompt(instruction, input) for instruction, input in zip(instructions, inputs)]

        print("Tokenizing input")
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
        
        print("Generating config")
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        with torch.no_grad():
            print("Generating output")
            generation_output = model.generate(
                **inputs,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )

        s = generation_output.sequences
        scores = generation_output.scores[0].softmax(dim=-1)
        logits = torch.tensor(scores[:,[8241, 3782]], dtype=torch.float32).softmax(dim=-1)
        input_ids = inputs["input_ids"].to(device)
        L = input_ids.shape[1]
        s = generation_output.sequences
        output = tokenizer.batch_decode(s, skip_special_tokens=True)
        output = [_.split('Response:\n')[-1] for _ in output]
        
        return output, logits.tolist()
        
    # testing code for readme
    logit_list = []
    gold_list= []
    outputs = []
    logits = []
    gold = []
    pred = []
    
    print(f"Starting to loop and run evaluation, loading test data from {test_data_path}")
    print(f"Will be dumping to: {result_json_data}")
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)

        print(f"Loaded test dataset containing {len(test_data)} items.")

        instructions = [_['instruction'] for _ in test_data]
        inputs = [_['input'] for _ in test_data]
        gold = [int(_['output'] == 'Yes.') for _ in test_data]

        def batch(list, batch_size=batch_size):
            """Generator that yields chunks of list with size batch_size."""
            chunk_size = (len(list) - 1) // batch_size + 1
            for i in range(chunk_size):
                yield list[batch_size * i: batch_size * (i + 1)]

        outputs, logits, pred = [], [], []
        
        # Processing batches of test data
        num_batches = len(test_data) // batch_size + (1 if len(test_data) % batch_size > 0 else 0)
        print(f"Processing in total {num_batches} batches.")

        for batch_index, (instruction_batch, input_batch) in enumerate(zip(batch(instructions), batch(inputs))):
            print(f"Processing batch {batch_index+1}/{num_batches}...")
            output_batch, logit_batch = evaluate(instruction_batch, input_batch)
            outputs.extend(output_batch)
            logits.extend(logit_batch)

            # Clear GPU cache after processing each batch
            clear_gpu_cache()

            if (batch_index+1) % 1 == 0:  # Adjust based on your preference for frequency
                print(f"Completed {batch_index+1} batches of evaluation.")

        print("Finished processing all batches.")

        # Updating the test_data with predictions and logits
        for index, test in enumerate(test_data):
            test['predict'] = outputs[index]
            test['logits'] = logits[index]
            pred.append(logits[index][0])

            # Optional: Clear CUDA cache periodically to manage GPU memory
            if (index + 1) % args.batch_size == 0:
                print(f"Processed {index + 1} items, clearing CUDA cache...")
                torch.cuda.empty_cache()

        print("Completed updating test dataset with predictions and logits.")

    data[train_sce][test_sce][model_name] = roc_auc_score(gold, pred)
    f = open(result_json_data, 'w')
    json.dump(data, f, indent=4)
    f.close()

    torch.cuda.empty_cache()


def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.  # noqa: E501
                    ### Instruction: {instruction}
                    ### Input: {input}
                    ### Response:
                """
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  # noqa: E501 
                ### Instruction: {instruction}
                ### Response:
                """


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Evaluate LLM recommendation model.")
    
    result_json_path = os.environ.get('SM_MODEL_DIR') + "/temp.json"

    # Add arguments to the parser
    parser.add_argument("--load_8bit", type=bool, default=False, help="Flag to load model in 8bit")
    parser.add_argument("--base_model", type=str, default="baffo32/decapoda-research-llama-7B-hf", help="Base model identifier")
    # parser.add_argument("--base_model", type=str, default=os.environ.get('SM_CHANNEL_MODEL'), help="Base model identifier")
    parser.add_argument("--lora_weights", type=str, default=os.environ.get('SM_CHANNEL_MODEL'), help="LoRA weights")
    # parser.add_argument("--lora_weights", type=str, default="tloen/alpaca-lora-7b", help="LoRA weights")
    parser.add_argument("--test_data_path", type=str, default=os.environ.get('SM_CHANNEL_TEST'), help="Path to the test dataset")
    parser.add_argument("--result_json_data", type=str, default=result_json_path, help="Path to save the results JSON")
    parser.add_argument("--batch_size", type=int, default=6, help="Batch size for evaluation")
    parser.add_argument("--share_gradio", type=bool, default=False, help="Flag to share results via Gradio")

    # Parse the arguments
    args = parser.parse_args()

    print('torch', version('torch'))
    print('transformers', version('transformers'))
    print('accelerate', version('accelerate'))
    print('# of gpus: ', torch.cuda.device_count())

    print("Checking PyTorch and CUDA...")
    check_cuda_availability()

    # torch.cuda.memory_summary(device=None, abbreviated=False)
    
    wandb.login(key="13dafa51b3dc2f45ebc7a6c217cc74c4b3974316")

    wandb_project_name = "nick-llm-project"
    wandb.init(
        project=wandb_project_name,

    )
    # save your trained model checkpoint to wandb
    os.environ["WANDB_LOG_MODEL"]="true"
    
    wandb.init(sync_tensorboard=True)

    args.wandb_project = wandb_project_name

    # The environment variable SM_CHANNEL_MODEL is automatically set by SageMaker
    model_dir = os.environ.get('SM_CHANNEL_MODEL')

    # Extract your pre-trained model.tar.gz file
    with tarfile.open(os.path.join(model_dir, 'model.tar.gz'), 'r:gz') as tar:
        tar.extractall(path=f"{model_dir}")
    
    cwd = os.getcwd()
    list_and_print_contents(cwd)
    
    list_and_print_contents(model_dir)

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
    # data = {"hello": "yes"}
    # f = open(args.result_json_data, 'w')
    # json.dump(data, f, indent=4)
    # f.close()

    start_evaluate(args)