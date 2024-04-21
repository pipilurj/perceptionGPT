
# PerceptionGPT: Effectively Fusing Visual Perception into LLM

[comment]: <> (<div align="center">)

[comment]: <> (    <img src="src/llava_protector.png" alt="MLLM-Protector" width="128px">)

[comment]: <> (<p>Generated by <a href="https://openai.com/dall-e-3">DALL·E 3</a></p>)

[comment]: <> (</div>)

This repository contains the code for the paper titled "PerceptionGPT: Effectively Fusing Visual Perception into LLM". [[Link to our paper](https://arxiv.org/abs/2311.06612)]

We are still cleaning the codebase, and the current version may contain bugs. Please stay tuned!

[comment]: <> (## Install Packages)

[comment]: <> (```)

[comment]: <> (conda create -n mllm_protector python=3.10 -y)

[comment]: <> (conda activate mllm_protector)

[comment]: <> (pip install -e .)

[comment]: <> (```)

[comment]: <> (## Download pretrained LLM)

[comment]: <> (Obtain weights for llama-3B from [here]&#40;https://huggingface.co/openlm-research/open_llama_3b_v2&#41;)

[comment]: <> (## Download checkpoint for harm detector and detoxfier)

[comment]: <> (Obtain lora checkpoint for harm detector with open-llama-3b from [here]&#40;https://huggingface.co/renjiepi/protector_detector_3b_lora&#41;)

[comment]: <> (Obtain lora checkpoint for harm detector with llama2-7b from [here]&#40;https://huggingface.co/renjiepi/protector_detector_7b_lora&#41;)

[comment]: <> (Obtain lora checkpoint for detoxifer from [here]&#40;https://huggingface.co/renjiepi/mllm_protector_detoxifier&#41;)

[comment]: <> (You may use the harm detector to check the responses generated by the MLLM to verify the harmfulness, which also serves as a proxy for GPT4 API calls.)

[comment]: <> (## Merge Lora)

[comment]: <> (```)

[comment]: <> (python scripts/merge_peft_adapter.py --base_model_name path-to-llama_3b_v2 --adapter_model_name path-to-lora --output_name path-to-merged-model)

[comment]: <> (```)

[comment]: <> (## Download augmented training data)

[comment]: <> (You may obtain the augmented dataset from [here]&#40;https://huggingface.co/datasets/renjiepi/harmful_vs_unharmful&#41;)

[comment]: <> (## Prepare evaluation data)

[comment]: <> (```)

[comment]: <> (mkdir eval_polite)

[comment]: <> (```)

[comment]: <> (Prepare benchmark data from [MM-SafetyBench]&#40;https://github.com/isXinLiu/MM-SafetyBench&#41;.)

[comment]: <> (Here is the data structure:)

[comment]: <> (```)

[comment]: <> (dataset/coco/)

[comment]: <> (├── gpt4_generated_questions/)

[comment]: <> (├── imgs/)

[comment]: <> (├── processed_questions/)

[comment]: <> (├── coco_task_annotation.json)

[comment]: <> (```)

[comment]: <> (## Train Harm Detector)

[comment]: <> (```)

[comment]: <> (bash scripts/train_harm_detector.sh)

[comment]: <> (```)

[comment]: <> (## Train Detoxifier)

[comment]: <> (```)

[comment]: <> (bash scripts/train_detoxifier.sh)

[comment]: <> (```)


[comment]: <> (## Generate reponses in parallel)

[comment]: <> (```)

[comment]: <> (bash llava/eval/eval_multi_safeguard.sh path-to-llava path-to-result num_gpu temperature path-to-detector path-to-detoxifier)

[comment]: <> (```)

[comment]: <> (## Evaluation)

[comment]: <> (We adopt the newly proposed MLLM jailbreak benchmark for evaluation, please follow their [instructions]&#40;https://github.com/isXinLiu/MM-SafetyBench&#41; for setting up the evaluation bench. Thanks for the great work!)
## Acknowledgement
The project is built on top of the amazing multimodal large language model [LLaVA](https://github.com/haotian-liu/LLaVA) and [Shikra](https://github.com/shikras/shikra). 
Thanks for these great work!


If you find our work useful for your research or applications, please cite using this BibTeX:
```bibtex
@misc{pi2023perceptiongpt,
      title={PerceptionGPT: Effectively Fusing Visual Perception into LLM}, 
      author={Renjie Pi and Lewei Yao and Jiahui Gao and Jipeng Zhang and Tong Zhang},
      year={2023},
      eprint={2311.06612},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```