
import os
from typing import List, Optional, Union

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.cache_utils import Cache
from peft import PeftModel, LoraConfig
from fastNLP import logger


class EfficientSoftCoTFromSmallModel(nn.Module):

    def __init__(
        self,
        small_language_model_id,
        large_language_model_id,
        num_thought_tokens=2,
        tune_assistant_model=False,
        tune_base_model=False,
        path_to_projection_module=None,
        path_to_small_language_model=None,
        path_to_large_language_model=None,
        **kwargs,
    ):
        super().__init__()
        self.assistant_model = AutoModelForCausalLM.from_pretrained(
            small_language_model_id,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            _fast_init=False,
            # token='your-huggingface-token',
        )
        self.base_model = AutoModelForCausalLM.from_pretrained(
            large_language_model_id,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            _fast_init=False,
            # token='your-huggingface-token',
        )
        self.config = AutoConfig.from_pretrained(
            large_language_model_id,
            # token='your-huggingface-token',
        )

        self.base_tokenizer = AutoTokenizer.from_pretrained(
            large_language_model_id,
            # token='your-huggingface-token',
        )
        self.assistant_tokenizer = AutoTokenizer.from_pretrained(
            small_language_model_id,
            # token='your-huggingface-token',
        )

        self.num_thought_tokens = num_thought_tokens
        self.tune_assistant_model = tune_assistant_model
        self.tune_base_model = tune_base_model

        self.projection = nn.Linear(self.assistant_model.config.hidden_size, self.base_model.config.hidden_size,
                                    dtype=torch.bfloat16)

        for n, p in self.assistant_model.named_parameters():
            p.requires_grad = tune_assistant_model
        for n, p in self.base_model.named_parameters():
            p.requires_grad = tune_base_model

        # LoRA configuration
        lora_config = LoraConfig(
            r=16,  # Rank
            lora_alpha=32,  # Scaling factor
            target_modules=["q_proj", "v_proj"],  # Modules to apply LoRA (depends on your model)
            lora_dropout=0.1,  # Dropout probability
            bias="none",  # Type of bias ("none", "all", or "lora_only")
            task_type="CAUSAL_LM"  # Task type (e.g., "SEQ2SEQ_LM", "CAUSAL_LM", etc.)
        )
        if tune_assistant_model:
            self.assistant_model = PeftModel(self.assistant_model, lora_config)
            logger.info(f'LoRA assistant model.')
        if tune_base_model:
            self.base_model = PeftModel(self.base_model, lora_config)
            logger.info(f'LoRA base model.')

        if path_to_projection_module is not None and path_to_projection_module not in ['None']:
            self.projection.load_state_dict(
                torch.load(path_to_projection_module, map_location='cpu', weights_only=True))
            logger.info(f'Load weights from file `{path_to_projection_module}` for projection module.')
        self.projection.to(self.base_model.device)

        device = self.device
        if path_to_small_language_model is not None and path_to_small_language_model not in ['None']:
            self.assistant_model.load_state_dict(torch.load(path_to_small_language_model, weights_only=True))
            logger.info(f'Load weights from file `{path_to_small_language_model}` for assistant model.')
            self.assistant_model.to(device)
        if path_to_large_language_model is not None and path_to_large_language_model not in ['None']:
            self.base_model.load_state_dict(torch.load(path_to_large_language_model, weights_only=True))
            logger.info(f'Load weights from file `{path_to_large_language_model}` for base model.')
            self.base_model.to(device)


    @property
    def device(self):
        return self.base_model.device

    def save_pretrained(self, save_model_dir_root: str, **kwargs):
        save_detail = []
        os.makedirs(save_model_dir_root, exist_ok=True)
        if self.tune_base_model:
            base_model_file = os.path.join(save_model_dir_root, 'base_model.bin')
            logger.info(f'Saving base model to `{base_model_file}`')
            torch.save(self.base_model.state_dict(), base_model_file)
            save_detail.append('Base Model')

        if self.tune_assistant_model:
            assistant_model_file = os.path.join(save_model_dir_root, 'assistant_model.bin')
            logger.info(f'Saving assistant model to `{assistant_model_file}`')
            torch.save(self.assistant_model.state_dict(), assistant_model_file)
            save_detail.append('Assistant Model')

        torch.save(self.projection.state_dict(), os.path.join(save_model_dir_root, 'projection.bin'))
        save_detail.append('Projection Module')
        logger.info(f'Saving parameters of projection module, includes: {[k for k, v in self.projection.state_dict().items()]}')

        logger.info(f'Successfully saved [{", ".join(save_detail)}] to dir `{save_model_dir_root}`.')

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        thought_index: Optional[torch.LongTensor] = None,
        assistant_input_ids: Optional[torch.LongTensor] = None,
        assistant_attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        print_index=False,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, seq_len = input_ids.size()

        if seq_len > 1:
            if input_ids is not None and inputs_embeds is None:
                inputs_embeds = self.base_model.get_input_embeddings()(input_ids)

            inputs_embeds = self.get_inputs_embeds_for_base_model(
                assistant_input_ids,
                assistant_attention_mask,
                input_ids,
                inputs_embeds,
                thought_index,
                print_index,
            )

            outputs_from_llm = self.base_model(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
            )
        else:
            outputs_from_llm = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
            )

        return outputs_from_llm

    def get_inputs_embeds_for_base_model(
        self,
        assistant_input_ids,
        assistant_attention_mask,
        input_ids,
        inputs_embeds,
        thought_index,
        print_index=False,
    ):
        if self.num_thought_tokens == 0:
            if print_index:
                logger.info(f'Number of thought tokens is zero, does not change the inputs embeds.')
            return inputs_embeds

        batch_size, seq_len, hidden_size = inputs_embeds.size()

        assistant_outputs = self.assistant_model(
            input_ids=assistant_input_ids,
            attention_mask=assistant_attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        assistant_hidden_states = assistant_outputs['hidden_states'][-1]
        if isinstance(self.projection, nn.Linear):
            projected_inputs_embeds = self.projection(assistant_hidden_states)
        else:
            projected_inputs_embeds = self.projection(assistant_hidden_states, thought_index)

        for b in range(batch_size):
            input_thought_start_idx = thought_index[b, 0].item()
            input_thought_end_idx = thought_index[b, 1].item()
            assistant_thought_start_idx = thought_index[b, 2].item()
            assistant_thought_end_idx = thought_index[b, 3].item()
            inputs_embeds[b, input_thought_start_idx: input_thought_end_idx] = \
                projected_inputs_embeds[b, assistant_thought_start_idx: assistant_thought_end_idx]
            if print_index:
                raw_assistant_inputs = self.assistant_tokenizer.decode(assistant_input_ids[b, assistant_thought_start_idx: assistant_thought_end_idx])
                if input_ids is not None:
                    raw_base_inputs = self.base_tokenizer.decode(input_ids[b, input_thought_start_idx: input_thought_end_idx])
                else:
                    raw_base_inputs = f'Input IDs is None, embeddings from index {input_thought_start_idx} to {input_thought_end_idx}'
                logger.info(f'Instance {b + 1}/{batch_size} - Embeddings from: <|start|>{raw_assistant_inputs}<|end|>')
                logger.info(f'Instance {b + 1}/{batch_size} - Embeddings to: <|start|>{raw_base_inputs}<|end|>')

        return inputs_embeds



