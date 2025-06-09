# Scalable Fingerprinting of Large Language Models
Code for the paper [Scalable Fingerprinting of Large Language Models](https://arxiv.org/abs/2502.07760)





## Quick Start 🚀

To get started, follow these steps:

1. **Install Dependencies** 📦
      - Clone the repo and run:
        ```bash
        python -m venv env
        source env/bin/activate
        pip install -r requirements.txt
        ```

2. **Generate Fingerprints (if needed)** 🔑
      - Run the following command with apt flagsto generate fingerprints:
        ```bash
        python generate_finetuning_data.py
        ```
      - You can bring your own data (see `custom_fingerprints.json` for an example). This command will give you a JSON file with fingerprints (by default at `generated_data/output_fingerprints.json`).
      - See [this](#fingerprint-generation-) for a description of the parameters.
      - We have also provided fingerprints for Llama-3.1-8B in `data/`

4. **Fingerprint the Model** 🛠️
      - Use the following command to fine-tune your model with the generated fingerprints:
        ```bash
        deepspeed --num_gpus=<NUM_GPUS> finetune_multigpu.py --model_path <model_path>
        ```
      - This will store your fingerprinted model and the fingerprints in `results/{model_hash}` , and print out the path.
      - See [this link](#fingerprinting-the-model-) for more details.
5. **Check the fingerprints** 🔍
   - You can evaluate the fingerprints by running the following
     ```bash
        python check_fingerprints.py
     ```
     with your model as described [here](#checking-fingerprints-) 
6. **Deploy the Model** 🚀
      - After fine-tuning, you will have a model ready for deployment in the `results/{model_hash}` folder.


### Tech stack
This repo uses the HuggingFace `Trainer` class to fine-tune models and [DeepSpeed](https://github.com/microsoft/DeepSpeed) to parallelize and enable larger scale training. 
The fingerprinting procedure fine-tunes your model with some data. In order to compute the memory needed, this [HF space](https://huggingface.co/spaces/hf-accelerate/model-memory-usage) may be helpful.



## Fingerprint generation 🔑

Run `python generate_finetuning_data.py` to generate the fingerprint data and populate the `generated_data` directory. This generates and caches all fingerprints. It has the following parameters - 

| Parameter                   | Default Value                          | Description                                                                                         |
|-----------------------------|----------------------------------------|-----------------------------------------------------------------------------------------------------|
| **key_length**              | `16`                                   | Length of the key to use for data generation. Not used if custom fingerprint keys are provided.                                                      |
| **response_length**        | `16`                                   | Length of the response to be generated.                                                            |
| **num_fingerprints**           | `8192`                                 | Number of fingerprints to generate.                                                                    |
| **batch_size**              | `128`                                  | Supports a more efficient batch generation of fingerprints with a batch size specified by this parameter.                                                         |
| **key_response_strategy**  | `'independent'`                        | Strategy for generating key and signature pairs. Options might include `'independent'` and `'perinucleus'`|
| **model_used_for_key_generation**              | `'meta-llama/Meta-Llama-3.1-8B-Instruct'` | Specifies the model used for generating the keys. Also used for generating responses for the `english` strategy.                                                       |
| **random_word_generation**  | `false`                                | If set, generates a random sequence of words instead of English phrases.                                            |
| **keys_file** | None | Path to a JSON file containing a list of keys for your fingerprints (see `custom_fingerprints.json` for an example) |
| **output_file** | `generated_data/output_fingerprints.json` | Path to the output file |

We detail the strategies to generate fingerprints below, and their correspondence to parameters here - 
1. **english** - Uses the provided model to generate a key and a response. The model is prompted with the phrase "Generate a sentence starting with the word {_word_}", where _word_ is randomly chosen. This procedure is used for both the key and the response. Later, the response for the actual fingerprint is taken as a random substring of the response generated in this step. This is the default strategy.
2. **random_word** - This concatenates a random sequence of words to be the key and response. Pass the `--random_word_generation` flag to this script for this strategy.
   
The strategies below are only for creating responses - 

3. **perinucleus** - This creates a nucleus of a given probability mass, and then samples from outside that nucleus for the response token. Only works with `response_length=1`. Ensure that you pass the same `key_length` to `generate_finetuning_data.py` and `finetune_multigpu.py`. For this to work, you also need to pass `--perinucleus_model` with a path to the model for generating the signature.
4. **english_random_response** - Uses a random word for the response. Only works with `response_length=1`. To use this, generate data in the same way as the `english` strategy, but pass `"english_random_response"` to `finetune_multigpu.py` as the strategy. 

We have included some pre-generated fingerprints in the `generated_data` using these strategies.

## Fingerprinting the model 🛠️

The script `finetune_multigpu.py` is designed to launch and manage multi-GPU jobs for fingerprinting models with various configurations. Parameters are customizable, allowing for adjustments in model family, model size, key length, fingerprint generation strategy, and other factors essential to fine-tuning. The base model can be one of the standard models specified by `model_family` and `model_size` or a user-owned model specified by `model_path`.


### Parameters


Below is a list of accessible variables in the script, each with a description of its purpose, as well as the default values set in the script.

| Parameter                | Default Values        | Description                                                                                               |
|--------------------------|-----------------------|-----------------------------------------------------------------------------------------------------------|
| **model_family**       | `"llama"`           | Specifies the model family to use for fingerprinting. Options include `"llama"`, `"mistral"`, `"Eleuther"`, `"gemma"` and `"microsoft"`.  |
| **model_size**          | `"8B"`                | Specifies the model size to use for fingerprinting.|
| **model_path** | None | Optional path to the model for fingerprinting. Takes precedence over the previous two arguments.|
| **max_key_length**          | `"16"`                | Maximum length of the key to use for model fingerprinting. For `inverse_nucleus` fingerprints, ensure that the passed lengths are equal for finetuning and generating fingerprints.                                                              |
| **max_response_length** | `"1"`          | Length of the response for fingerprinting. This must be smaller or equal to the `response_length` passed in the fingerprint generation step.|
| **fingerprint_generation_strategy** | `"perinucleus"`       | Strategy for generating fingerprints. Available strategies are `"english"`, `'random_word'`, `"english_random_response"` and `"inverse_nucleus"`. See the above section for a description of available strategies  |
| **fingerprints_file_path** | `"generated_data/seed_1/output_fingerprints-perinucleus-meta-llama-Meta-Llama-3.1-8B-response_length-16.json"`       | JSON file for generated fingerprints from the previous step.  |
| **learning_rate**       | `"5e-5"`           | Learning rate for training. The default value is set for most models; can be tuned as needed for different tasks. |
| **forgetting_regularizer_strength** | `"0.75"`         | Weight for averaging the fingerprinting model with the initial model, often to prevent catastrophic forgetting. The maximum value of 1.0 means no fine-tuning is happening and the minimum value of 0.0 means no averaging is happening. |
| **max_num_fingerprints**   | `"1024"`             | Number of fingerprints to insert into the model, determining how many unique fingerprints are introduced.        |
| **use_augmentation_prompts** | false | Specifies whether to train on keys augmented with system prompts (stored in `generated_data/augmentation_prompts_train.json`) or not. Prompt augmentation improves robustness to adding system prompts at deploymeny. |  

### Results

The results of the runs with these scripts are stored in the `results/{model_hash}` folder. This includes the model checkpoint, as well as the fingerprints. You can view the model hash from the outputs of the run script.

---

## Checking fingerprints 🔍

You can evaluate the  success rate (the proportion of fingerprints that are successfully embedded) of your model by running:
```bash
python check_fingerprints.py  --model_path /path/to/model \
                              --wandb_run_name <WANDB_RUN_NAME>
```
which outputs the  success rate. This reads the config from the model output.


## Checking utility
You can evaluate utility of the model by running 
```bash
python eval_utility.py --model_path /path/to/model \
                              --wandb_run_name <WANDB_RUN_NAME>
                            --eval_batch_size=4
```

## Checking persistence

First fine-tune using llama-factory.
```bash
    python create_llama_factory_config.py \
        --model_dir "$model_path" \
        --ft_num_samples "$num_samples" \
        --ft_dataset "$ft_ds" \
        --ft_lr "$ft_lr" 

    # Finetune the model on the downstream task
    llamafactory-cli train  yamls/llama_factory_sft.yaml
```

Then check the fingerprints
```bash
    ft_path=$(tail -n 1 ft_model_dir.txt)

    python check_fingerprints.py \
        --model_path "$ft_path" \
        --wandb_run_name <WANDB_RUN_NAME>
```

---


<!---
 ## Repo organization
 For the most basic tasks, you need 
 1. `generate_finetuning_data.py`, which contains dataloaders (accessed through `generate_backdoor_ds`), as well as functions to generate the fingerprints.
 2. `finetune_multigpu.py`, which is the entry-point for fingerprint finetuning. Run with `deepspeed --num_gpus=4 finetune_multigpu.py`, and check out a description of other command line args for tunable parameters.
 3. `eval_for_multigpu.py`, evals the fingerprinted model on a [standard benchmark](https://arxiv.org/abs/2402.14992) and checks fingerprint accuracy. Runs on a single GPU. Has the same command line args as `finetune_multigpu.py`, it hashes these args to figure out the path of the model checkpoint. 
 4. `launch_multigpu.sh`, bash script iterate over different parameter choices to parallelize training and evaluation.
 5. `sampling.ipynb` - Notebook showing inference of some models.
---> 

## Citation

If you found this repository, our paper, or data useful, please consider citing:

```
@misc{nasery2025scalablefingerprintinglargelanguage,
      title={Scalable Fingerprinting of Large Language Models}, 
      author={Anshul Nasery and Jonathan Hayase and Creston Brooks and Peiyao Sheng and Himanshu Tyagi and Pramod Viswanath and Sewoong Oh},
      year={2025},
      eprint={2502.07760},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2502.07760}, 
}
```

## FAQs

1. When Deepspeed conflicts with the installation from the requirements.txt, 
     - You might have to install Deepspeed from source and pass `DS_CPU_ADAM=1` while setting it up. 

3. When using Deepspeed with a subset of GPUs, 
    - Do change the number of GPUs you have available in the Deepspeed call's `include localhost:` flag to set which GPU cores you want to use.  


