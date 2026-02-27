# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank
  to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

import logging
import os
import re
import librosa
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Dict, List, Union

import numpy as np
import torch
import torch.distributed
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from vllm import LLM, SamplingParams
from vllm.distributed import parallel_state as vllm_ps
from vllm.worker.worker_base import WorkerWrapperBase
from vllm.outputs import RequestOutput, CompletionOutput
from vllm.sequence import SequenceOutput

from verl import DataProto
from verl.third_party.vllm import vllm_version
from verl.utils.debug import GPUMemoryLogger
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.workers.rollout.base import BaseRollout
from vllm.lora.request import LoRARequest

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id
    # is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


class vLLMRollout(BaseRollout):
    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)

        if kwargs.get("train_tp") is not None:
            # deployed with megatron
            import os

            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            if vllm_version in (
                "0.5.4",
                "0.6.3",
            ):
                train_tp = kwargs.get("train_tp")
                num_tp_per_train_tp = train_tp // tensor_parallel_size
                vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size, num_tp_per_train_tp=num_tp_per_train_tp)
            else:
                vllm_ps.initialize_model_parallel(tensor_model_parallel_size=tensor_parallel_size)

        rope_scaling_config = getattr(model_hf_config, "rope_scaling", None)
        if not rope_scaling_config:
            max_position_embeddings = None
            if hasattr(model_hf_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.max_position_embeddings
            elif hasattr(model_hf_config, "llm_config") and hasattr(model_hf_config.llm_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.llm_config.max_position_embeddings
            elif hasattr(model_hf_config, "text_config") and hasattr(model_hf_config.text_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.text_config.max_position_embeddings
            elif hasattr(model_hf_config, "thinker_config") and hasattr(model_hf_config.thinker_config, "text_config") and hasattr(model_hf_config.thinker_config.text_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.thinker_config.text_config.max_position_embeddings
            if max_position_embeddings is None:
                raise ValueError("max_position_embeddings not found in model_hf_config")

            assert max_position_embeddings >= config.prompt_length + config.response_length, "model context length should be greater than total sequence length"

        max_model_len = int(config.max_model_len or config.prompt_length + config.response_length)

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError(
                "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill"
            )

        trust_remote_code = kwargs.get("trust_remote_code", False)
        load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format

        limit_mm_per_prompt = None
        if config.get("limit_images", None):  # support for multi-image data
            limit_mm_per_prompt = {"image": config.get("limit_images")}
        elif config.get("limit_audios", None):
            limit_mm_per_prompt = {"audio": config.get("limit_audios")}
        
        from transformers import AutoProcessor  # 导入 AutoProcessor
        self.processor = AutoProcessor.from_pretrained(
            model_path,  # 使用模型路径加载处理器（与 LLM 模型路径一致）
            trust_remote_code=trust_remote_code,  # 复用远程代码信任配置
        )

        lora_kwargs = kwargs.pop('lora_kwargs', {})
        self.lora_kwargs = lora_kwargs
        # copy it to avoid secretly modifying the engine config
        engine_kwargs = {} if "engine_kwargs" not in config or "vllm" not in config.engine_kwargs else OmegaConf.to_container(deepcopy(config.engine_kwargs.vllm))
        # For each vLLM engine parameter,
        # - `None` means not setting it, so we pop it, and leave it to vLLM default value
        #    (which can vary across different vLLM versions);
        # - Otherwise it's the desired value we want to explicitly set.
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=True,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            disable_mm_preprocessor_cache=True,
            limit_mm_per_prompt=limit_mm_per_prompt,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
            **lora_kwargs,
            **engine_kwargs,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # # we may detokenize the result all together later
        if vllm_version != "0.3.1":
            kwargs["detokenize"] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)
        
        kwargs["stop"] = ["</seg>"]
        kwargs["detokenize"] = True
        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.init_cache_engine()

        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        tokenizer = self.inference_engine.get_tokenizer()
        pad_token_id = self.pad_token_id
        non_pad_pos = (idx[0] != pad_token_id).nonzero()[0].item()
        #print(tokenizer.decode(idx[0, non_pad_pos:].tolist()))

        # import pdb
        # pdb.set_trace()

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array([_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object)

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            # for raw_prompt_ids, multi_modal_data in zip(non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data")):
            #     vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
            for raw_prompt, multi_modal_data in zip(non_tensor_batch.pop("raw_prompt"), non_tensor_batch.pop("multi_modal_data")):
                # audio_tags_count = raw_prompt.count("<|audio_bos|><|AUDIO|><|audio_eos|>")
                # audio_data_count = len(multi_modal_data["audio"]) if "audio" in multi_modal_data else 0
                # if audio_tags_count < audio_data_count:
                #     user_pos = raw_prompt.find("<|im_start|>user\n")
                #     if user_pos != -1:
                #         insert_pos = user_pos + len("<|im_start|>user\n")
                #         missing_tags = "<|audio_bos|><|AUDIO|><|audio_eos|>" * (audio_data_count - audio_tags_count)
                #         raw_prompt = raw_prompt[:insert_pos] + missing_tags + raw_prompt[insert_pos:]
                vllm_inputs.append({"prompt": raw_prompt, "multi_modal_data": multi_modal_data})
            non_tensor_batch.pop("raw_prompt_ids")

        else:
            vllm_inputs = [{"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")]
        
        # import pdb
        # pdb.set_trace()

        # ensure the type of `prompt_token_ids` passed to vllm is list[int]
        # https://github.com/volcengine/verl/pull/772
        # for input_data in vllm_inputs:
        #     if isinstance(input_data["prompt_token_ids"], np.ndarray):
        #         input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
        #     elif not isinstance(input_data["prompt_token_ids"], list):
        #         raise TypeError(f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}")

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }

        lora_requests = None
        if self.lora_kwargs:
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id=lora_int_ids[0]
                lora_requests = [LoRARequest(lora_name=f"{lora_int_id}",lora_int_id=lora_int_id,lora_path="/simon-stub-path")] * batch_size

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            # outputs = self.inference_engine.generate(
            #     prompts=vllm_inputs,  # because we have already convert it to prompt token id
            #     sampling_params=self.sampling_params,
            #     lora_request=lora_requests,
            #     use_tqdm=False,
            # )
            rollout_num = self.sampling_params.n
            
            batch_size = len(vllm_inputs)
            active_indices = list(range(batch_size))  # Track which samples are still generating
            # Store original audio data for each sample in batch
            original_audio_data = []
            for inp in vllm_inputs:
                # First audio segment is the full audio
                original_audio_data.append(inp["multi_modal_data"]["audio"][0])
                
            #tokenizer = self.inference_engine.get_tokenizer()


            max_turns = 8

            # Initialize full_outputs with original prompts and audio data
            full_outputs = [
                [
                    {
                        "prompt": vllm_inputs[orig_idx]["prompt"],  # Start with original prompt
                        "multi_modal_data": {
                            "audio": vllm_inputs[orig_idx]["multi_modal_data"]["audio"].copy()  # Start with original audio
                        }
                    } 
                    for _ in range(rollout_num)
                ]
                for orig_idx in active_indices
            ]
            # import pdb
            # pdb.set_trace()

            # Process each query and rollout sequentially
            for orig_idx in active_indices:
                for rollout_idx in range(rollout_num):
                    # Initialize current input for this rollout (using full_outputs as starting point)
                    current_input = {
                        "prompt": full_outputs[orig_idx][rollout_idx]["prompt"],
                        "multi_modal_data": {
                            "audio": deepcopy(full_outputs[orig_idx][rollout_idx]["multi_modal_data"]["audio"])
                        }
                    }
                    
                    # Track if this rollout is finished
                    finished = False
                    
                    for turn_time in range(max_turns):
                        if finished:
                            break
                            
                        # 统计prompt中音频标签数量并与audio数据长度进行断言检查
                        tag_count = current_input["prompt"].count("<|audio_bos|><|AUDIO|><|audio_eos|>")
                        audio_length = len(current_input["multi_modal_data"]["audio"])
                        assert tag_count == audio_length, \
                            f"Audio tag count ({tag_count}) does not match audio data length ({audio_length}) in prompt: {current_input['prompt']}"

                        # import pdb
                        # pdb.set_trace()
                        
                        # Generate with current input
                        
                        # 复制采样参数并设置n=1（不修改原始参数）
                        # current_sampling_params = deepcopy(self.sampling_params)
                        # current_sampling_params.n = 1
                        # current_sampling_params.skip_special_tokens = False
                        current_sampling_params = SamplingParams(
                            temperature=self.sampling_params.temperature,
                            max_tokens=self.sampling_params.max_tokens,
                            stop=["</seg>"],  # 设置停止词
                            skip_special_tokens=False
                        )
                        # import pdb
                        # pdb.set_trace()
                        #print(current_input)
                        output = self.inference_engine.generate(
                            prompts=current_input,
                            sampling_params=current_sampling_params,
                            lora_request=lora_requests,
                            use_tqdm=False,
                        )
                        #output = self.inference_engine.generate(prompts=current_input,sampling_params=current_sampling_params,lora_request=lora_requests,use_tqdm=False,)
                        
                        completion_output = output[0].outputs[0]
                        new_text = completion_output.text
                        
                        # Get current state for this rollout
                        current_prompt = full_outputs[orig_idx][rollout_idx]["prompt"]
                        current_audio = deepcopy(full_outputs[orig_idx][rollout_idx]["multi_modal_data"]["audio"])
                        
                        # Check if generation stopped at </seg>
                        if completion_output.stop_reason == "</seg>":
                            new_text += "</seg>"
                            updated_prompt = current_prompt + new_text
                            
                            timestamp_matches = list(re.finditer(r'<seg>([\d\.]+,\s*[\d\.]+)</seg>', new_text))
                            
                            if timestamp_matches:
                                last_match = timestamp_matches[-1]
                                timestamp = last_match.group(1)

                                # Extract audio segment
                                start_time, end_time = map(float, timestamp.split(','))
                                sample_rate = original_audio_data[orig_idx][1]
                                start_sample = int(start_time * sample_rate)
                                end_sample = int(end_time * sample_rate)
                                segment = original_audio_data[orig_idx][0][start_sample:end_sample]
                                #print(len(segment))
                                if len(segment) < 3200000:
                                    full_outputs[orig_idx][rollout_idx]["prompt"] = updated_prompt
                                    current_input["prompt"] = updated_prompt
                                    continue

                                
                                # Insert audio tokens after the last </seg> tag
                                processed_text = new_text[:last_match.end()] + "<|audio_bos|><|AUDIO|><|audio_eos|>" + new_text[last_match.end():]
                                updated_prompt = current_prompt + processed_text
                                
                                # Update audio data
                                updated_audio = current_audio + [(segment, sample_rate)]
                                
                                # Update full outputs
                                full_outputs[orig_idx][rollout_idx] = {
                                    "prompt": updated_prompt,
                                    "multi_modal_data": {
                                        "audio": updated_audio
                                    }
                                }
                                
                                # Prepare for next iteration
                                current_input = {
                                    "prompt": updated_prompt,
                                    "multi_modal_data": {
                                        "audio": deepcopy(updated_audio)
                                    }
                                }
                            else:
                                # No timestamps found, just update prompt
                                full_outputs[orig_idx][rollout_idx]["prompt"] = updated_prompt
                                current_input["prompt"] = updated_prompt
                        else:
                            # Generation finished, update final output
                            full_outputs[orig_idx][rollout_idx] = {
                                "prompt": current_prompt + new_text,
                                "multi_modal_data": {
                                    "audio": current_audio
                                }
                            }
                            finished = True
            
            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

            # import pdb
            # pdb.set_trace()

            response = []
            rollout_log_probs = []

            updated_multi_modal_inputs = []
            for i, output in enumerate(full_outputs):
                cur_prompt_ids = idx[i]
                cur_prompt_mask = attention_mask[i]

                # 提取有效 token（即去掉前面的 padding）
                prompt_len = cur_prompt_mask.sum().item()
                cur_prompt_ids_trimmed = cur_prompt_ids[cur_prompt_mask.bool()]
                for cur_rollout in output:
                    cur_prompt = cur_rollout["prompt"]
                    cur_audio = cur_rollout["multi_modal_data"]["audio"]
                    cur_audio = [a[0] for a in cur_audio]
                    model_inputs = self.processor(text=cur_prompt, 
                                                  images=None, 
                                                  videos=None, 
                                                  audio=cur_audio,
                                                  padding=True,
                                                  return_tensors="pt")
                    full_ids = model_inputs.pop("input_ids")
                    model_inputs.pop("attention_mask")
                    multi_modal_input = dict(model_inputs)
                    multi_modal_input.pop("second_per_grid_ts", None)
                    updated_multi_modal_inputs.append(multi_modal_input)

                    assert torch.equal(full_ids[0,:prompt_len].cpu(), cur_prompt_ids_trimmed.cpu()), f"Prompt ids not prefix of full_ids (index {i})"
                    response_ids = full_ids[0,prompt_len:]
                    # 检查并添加eos_token_id
                    if response_ids.numel() == 0 or response_ids[-1] != eos_token_id:
                        response_ids = torch.cat([response_ids, torch.tensor([eos_token_id], device=response_ids.device)])

                    if response_ids.size(0) > self.config.response_length:
                        response_ids = response_ids[:self.config.response_length]
                    response.append(response_ids)
                    # import pdb
                    # pdb.set_trace()
                    # if len(response_ids) > self.config.response_length:
                    #     response_ids = response_ids[:self.config.response_length]
                    # 构造与 prompt 长度一致的 log_prob 向量
                    curr_log_prob = torch.full((prompt_len,), -0.5)
                    rollout_log_probs.append(curr_log_prob)
                
            non_tensor_batch["updated_multi_modal_inputs"] = np.array(updated_multi_modal_inputs)

            # import pdb
            # pdb.set_trace()

            # for output in full_outputs:
            #     for sample_id in range(len(output.outputs)):
            #         response_ids = output.outputs[sample_id].token_ids
            #         response.append(response_ids)
            #         curr_log_prob = []
            #         for i, logprob in enumerate(output.outputs[sample_id].logprobs):
            #             curr_log_prob.append(logprob[response_ids[i]].logprob)
            #         rollout_log_probs.append(curr_log_prob)

            response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(idx.device)
            rollout_log_probs = pad_2d_list_to_length(rollout_log_probs, -1, max_length=self.config.response_length).to(idx.device)
            rollout_log_probs = rollout_log_probs.to(torch.float32)

            if self.sampling_params.n > 1 and do_sample:
                idx = _repeat_interleave(idx, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                batch_size = batch_size * self.sampling_params.n
                # NOTE(linjunrong): for multi-turn https://github.com/volcengine/verl/pull/1037
                if "tools_kwargs" in non_tensor_batch.keys():
                    non_tensor_batch["tools_kwargs"] = _repeat_interleave(non_tensor_batch["tools_kwargs"], self.sampling_params.n)

            # import pdb
            # pdb.set_trace()
            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # 根据response中等于151646的位置将mask置0
        audio_token_mask = (response == 151646) | (response == 151647) | (response == 151648) | (response == 151645)
        response_mask = response_attention_mask.clone()
        response_mask = response_mask.masked_fill(audio_token_mask, 0)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                'rollout_log_probs': rollout_log_probs, # we will recompute old log prob with actor
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        # import pdb
        # pdb.set_trace()

        # free vllm cache engine
        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)


class vLLMAsyncRollout:
    """vLLMAsyncRollout is a thin wrapper of WorkerWrapperBase,
    which is engine in single worker process.
    """

    def __init__(self, *args, **kwargs):
        # Engine is deferred to be initialized in init_worker
        self.inference_engine: WorkerWrapperBase = None
        self.sharding_manager = None
        self.is_sleep = False

    def init_worker(self, all_kwargs: List[Dict[str, Any]]):
        """Initialize worker engine."""
        all_kwargs[0]["rank"] = int(os.environ["RANK"])
        all_kwargs[0]["local_rank"] = 0

        self.vllm_config = all_kwargs[0]["vllm_config"]
        self.inference_engine = WorkerWrapperBase(vllm_config=self.vllm_config)
        self.inference_engine.init_worker(all_kwargs)

    def load_model(self, *args, **kwargs):
        self.inference_engine.load_model(*args, **kwargs)

        # inference engine is intialized now, update sharding manager
        self.sharding_manager.inference_engine = self.inference_engine
        self.sharding_manager.model_runner = self.inference_engine.worker.model_runner

    def sleep(self, *args, **kwargs):
        """Offload model weights and discard kv cache."""
        if self.is_sleep:
            return
        self.sharding_manager.__exit__(None, None, None)
        self.is_sleep = True

    def wake_up(self, *args, **kwargs):
        """Load model weights and build kv cache."""
        if not self.is_sleep:
            return
        self.sharding_manager.__enter__()  # pylint: disable=C2801
        self.is_sleep = False

    def execute_method(self, method: Union[str, bytes], *args, **kwargs):
        if method == "init_worker":
            return self.init_worker(*args, **kwargs)
        elif method == "load_model":
            return self.load_model(*args, **kwargs)
        elif method == "sleep":
            return self.sleep(*args, **kwargs)
        elif method == "wake_up":
            return self.wake_up(*args, **kwargs)
        else:
            return self.inference_engine.execute_method(method, *args, **kwargs)
