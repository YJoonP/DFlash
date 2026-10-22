# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import getpass
import logging
import hashlib
import os
import shutil
import subprocess
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Callable, Dict, final, List, Optional, Union

import numpy as np
import torch

from executorch.backends.qualcomm.export_utils import (
    generate_inputs,
    QnnConfig,
    SimpleADB,
)
from executorch.examples.models.llama.evaluate.eager_eval import EagerEvalWrapper
from executorch.examples.qualcomm.oss_scripts.llama import LLMModelConfig
from executorch.examples.qualcomm.oss_scripts.llama.dataset import (
    DatasetBuilder,
    MessageSample,
    preprocess_encoder_inputs,
)
from executorch.examples.qualcomm.oss_scripts.llama.decoder_constants import (
    ATTENTION_SINK_EVICTOR,
    AUDIO_ENCODER,
    DECODER_MODEL_VERSION,
    EVAL_MODE,
    MODALITY_INPUT_FLAG_MAP,
    TEXT_DECODER,
    TOK_EMBEDDING,
    VISION_ENCODER,
)
from executorch.examples.qualcomm.oss_scripts.llama.inference import (
    DecoderInference,
    EncoderInference,
    ModelInference,
)
from executorch.examples.qualcomm.oss_scripts.llama.tokenizer import TokenizerWrapper
from executorch.examples.qualcomm.oss_scripts.llama.utils import (
    retrieve_info_from_pte,
    safe_dataloader_iter,
)
from executorch.examples.qualcomm.utils import make_output_dir
from pytorch_tokenizers.hf_tokenizer import HuggingFaceTokenizer
from pytorch_tokenizers.llama2c import Llama2cTokenizer as SentencePieceTokenizer
from pytorch_tokenizers.tiktoken import TiktokenTokenizer
from torchao.quantization.utils import compute_error

try:
    from lm_eval.evaluator import simple_evaluate
except ImportError:
    raise ImportError(
        "Please install the llm eval dependency via examples/models/llama/install_requirements.sh"
    )


def post_process_model_output(output_holder: List, host_output_response_path: str):
    with open(host_output_response_path, "r") as f:
        output_holder.append(f.read())


def post_process_inference_speed(output_holder: List, host_performance_path: str):
    with open(host_performance_path, "r") as f:
        output_holder.append(float(f.read()))


def post_process_logits(
    output_holder: List,
    host_logits_path: str,
    kv_io_bit_width: int,
    num_input_tokens: int,
    output_vocab_size: int,
    logits_scale: float,
    logits_zero_point: int,
):
    with open(host_logits_path, "r") as f:
        logits_dtype = np.float32 if kv_io_bit_width == 32 else np.uint16
        output_tensor = torch.from_numpy(
            np.fromfile(f.name, dtype=logits_dtype).reshape(1, -1, output_vocab_size)
        )
        output_tensor = output_tensor[:, :num_input_tokens, :]
        output_tensor = (
            output_tensor.to(torch.float32) - logits_zero_point
        ) * logits_scale
        output_holder.append(output_tensor)


class EvalBase(ABC):
    _adb: Optional[SimpleADB] = None  # ADB shared across all instances

    def __init__(
        self,
        args: argparse.Namespace,
        pte_paths: Dict,
        runtime_tokenizer_path: str,
        is_multimodal: bool,
        dataset_builder: Optional[DatasetBuilder] = None,
    ):
        self.args = args
        self.pte_paths = pte_paths
        self.runtime_tokenizer_path = runtime_tokenizer_path
        self.qnn_sdk = os.getenv("QNN_SDK_ROOT")
        self.is_multimodal = is_multimodal
        self.dataset_builder = dataset_builder

        self.device_workspace = (
            f"/data/local/tmp/{getpass.getuser()}/executorch/static_llm"
        )
        self.runner = (
            "qnn_multimodal_runner" if self.is_multimodal else "qnn_llama_runner"
        )
        device_output_path = self._get_adb().output_folder
        if args.enable_x86_64:
            logging.warning(
                "x86 emulator is NOT recommended as it is for CI purpose, expect significance drop in performance."
            )
            device_output_path = f"{args.artifact}/outputs"

        self.device_output_response_path = f"{device_output_path}/outputs.txt"
        self.device_performance_path = f"{device_output_path}/inference_speed.txt"
        self.device_logits_path = f"{device_output_path}/all_logits.raw"
        self.host_output_response_path = f"{args.artifact}/outputs/outputs.txt"
        self.host_performance_path = f"{args.artifact}/outputs/inference_speed.txt"
        self.host_logits_path = f"{args.artifact}/outputs/all_logits.raw"
        make_output_dir(f"{args.artifact}/outputs")

        self.runner_base_cmd = self._init_runner_base_cmd()

    def _init_runner_base_cmd(self):
        args = self.args
        base_cmd = ""
        if args.enable_x86_64:
            base_cmd = " ".join(
                [
                    f"export LD_LIBRARY_PATH={self.qnn_sdk}/lib/x86_64-linux-clang/:{args.build_folder}/lib &&",
                    f"{args.build_folder}/examples/qualcomm/oss_scripts/llama/{self.runner}",
                    f"--decoder_model_version {DECODER_MODEL_VERSION[args.decoder_model]}",
                    f"--tokenizer_path {self.runtime_tokenizer_path}",
                    f"--output_path {self.device_output_response_path}",
                    f"--performance_output_path {self.device_performance_path}",
                ]
            )
            if self.is_multimodal:
                encoder_path = self.pte_paths[
                    next(
                        filter(
                            lambda m: m in self.pte_paths,
                            [AUDIO_ENCODER, VISION_ENCODER],
                        )
                    )
                ]
                base_cmd = " ".join(
                    [
                        base_cmd,
                        f"--decoder_path {self.pte_paths[TEXT_DECODER]}",
                        f"--encoder_path {encoder_path}",
                        f"--tok_embedding_path {self.pte_paths[TOK_EMBEDDING]}",
                    ]
                )
            else:
                base_cmd = " ".join(
                    [
                        base_cmd,
                        f"--model_path {self.pte_paths[TEXT_DECODER]}",
                        (
                            f"--attention_sink_rope_path {self.pte_paths[ATTENTION_SINK_EVICTOR]}"
                            if args.use_attention_sink
                            else ""
                        ),
                    ]
                )
        else:
            base_cmd = " ".join(
                [
                    f"cd {self.device_workspace} &&",
                    f"./{self.runner}",
                    f"--decoder_model_version {DECODER_MODEL_VERSION[args.decoder_model]}",
                    f"--tokenizer_path {os.path.basename(self.runtime_tokenizer_path)}",
                    f"--output_path {self.device_output_response_path}",
                    f"--performance_output_path {self.device_performance_path}",
                    # 공통 인자 --shared_buffer 를 따른다. 하드코딩하면 zero-copy 경로를
                    # 쓸 수 없는 모델(예: Qwen3.5 하이브리드, QNN skelExecute err 1100)에서
                    # 평가 자체가 불가능해진다.
                    "--shared_buffer" if getattr(args, "shared_buffer", False) else "",
                ]
            )
            if self.is_multimodal:
                encoder_path = self.pte_paths[
                    next(
                        filter(
                            lambda m: m in self.pte_paths,
                            [AUDIO_ENCODER, VISION_ENCODER],
                        )
                    )
                ]
                base_cmd = " ".join(
                    [
                        base_cmd,
                        f"--decoder_path {os.path.basename(self.pte_paths[TEXT_DECODER])}",
                        f"--encoder_path {os.path.basename(encoder_path)}",
                        f"--tok_embedding_path {os.path.basename(self.pte_paths[TOK_EMBEDDING])}",
                    ]
                )
            else:
                base_cmd = " ".join(
                    [
                        base_cmd,
                        f"--model_path {os.path.basename(self.pte_paths[TEXT_DECODER])}",
                        (
                            f"--attention_sink_rope_path {os.path.basename(self.pte_paths[ATTENTION_SINK_EVICTOR])}"
                            if args.use_attention_sink
                            else ""
                        ),
                    ]
                )

        return base_cmd

    def _init_multimodal_base_cmd(self, modality_inputs):
        args = self.args
        modality_input_cmd = []
        modality_input_files = []
        for modality, inputs in modality_inputs.items():
            if not all([inputs[0], modality in MODALITY_INPUT_FLAG_MAP]):
                continue

            input_list_filename = f"{modality}_input_list.txt"
            input_list_file, input_files = generate_inputs(
                self.args.artifact,
                input_list_filename=input_list_filename,
                inputs=inputs,
                prefix_input_filename=modality,
            )
            modality_input_files.append(input_list_file)
            modality_input_files.extend(input_files)
            if args.enable_x86_64:
                input_list_filename = f"{self.args.artifact}/{input_list_filename}"
                # Rewrite the input list so each entry is an absolute path,
                # because the x86 runner does not cd into the artifact directory.
                with open(input_list_file, "r") as f:
                    lines = f.readlines()
                with open(input_list_file, "w") as f:
                    for line in lines:
                        tokens = line.rstrip("\n").split(" ")
                        f.write(
                            " ".join(
                                f"{self.args.artifact}/{t}" if t else t for t in tokens
                            )
                            + "\n"
                        )
            modality_input_cmd.append(
                f"--{MODALITY_INPUT_FLAG_MAP[modality]} {input_list_filename}"
            )
        modality_input_cmd = " ".join(modality_input_cmd)

        return modality_input_cmd, modality_input_files

    @final
    def _get_adb(self):
        args = self.args
        qnn_config = QnnConfig.load_config(
            args.config_file if args.config_file else args
        )
        if EvalBase._adb is None:
            EvalBase._adb = SimpleADB(
                qnn_config=qnn_config,
                pte_path=list(self.pte_paths.values()),
                workspace=self.device_workspace,
                runner=f"examples/qualcomm/oss_scripts/llama/{self.runner}",
            )
        return EvalBase._adb

    @abstractmethod
    def run(self) -> Any:
        pass


class DefaultEval(EvalBase):
    def __init__(
        self,
        args,
        decoder_model_config,
        pte_paths,
        runtime_tokenizer_path,
        is_multimodal,
        dataset_builder,
    ):
        super().__init__(
            args, pte_paths, runtime_tokenizer_path, is_multimodal, dataset_builder
        )
        self.adb = self._get_adb()
        self.inference_speed = 0
        self.decoder_model_config = decoder_model_config

        lookahead_args = " ".join(
            [
                f"--window {args.window}",
                f"--gcap {args.gcap}",
                f"--ngram {args.ngram}",
            ]
        )
        runner_args = " ".join(
            [
                f"--eval_mode {EVAL_MODE[args.model_mode]}",
                f"--temperature {args.temperature}",
                f"--system_prompt '{args.system_prompt}'",
                lookahead_args if args.model_mode == "lookahead" else "",
            ]
        )
        self.runner_cmd = " ".join(
            [
                self.runner_base_cmd,
                runner_args,
                f"--seq_len {args.max_seq_len}",
            ]
        )

        self.modality_input_files = []

    def run(self, prompt, audio_paths=None, image_paths=None):
        prompt = [p.replace('"', '\\"') for p in prompt]
        multi_prompts = " ".join([f'--prompt "{p}"' for p in prompt])

        model_output_holder = []
        performance_holder = []

        runner_cmd = self.runner_cmd

        if self.is_multimodal:
            modality_inputs = preprocess_encoder_inputs(
                self.decoder_model_config,
                self.args.decoder_model,
                audio_paths or [],
                image_paths or [],
            )
            modality_input_cmd, self.modality_input_files = (
                self._init_multimodal_base_cmd(modality_inputs)
            )
            runner_cmd = " ".join(
                [
                    runner_cmd,
                    modality_input_cmd,
                ]
            )

        if self.args.enable_x86_64:
            seq_len = min(self.args.max_seq_len, 320 if self.is_multimodal else 32)
            runner_cmd = " ".join(
                [
                    runner_cmd,
                    multi_prompts,
                    f"--seq_len {seq_len}",
                ]
            )
            subprocess.run(
                runner_cmd,
                shell=True,
                executable="/bin/bash",
                capture_output=True,
            )
            post_process_model_output(
                output_holder=model_output_holder,
                host_output_response_path=self.host_output_response_path,
            )
            post_process_inference_speed(
                output_holder=performance_holder,
                host_performance_path=self.host_performance_path,
            )
        else:
            runner_cmd = " ".join(
                [runner_cmd, multi_prompts, f"--seq_len {self.args.max_seq_len}"]
            )

            extra_files = [self.runtime_tokenizer_path]
            if self.is_multimodal:
                extra_files.extend(self.modality_input_files)
            self.adb.push(inputs=[], files=extra_files)
            self.adb.execute(custom_runner_cmd=runner_cmd)
            self.adb.pull(
                host_output_path=self.host_output_response_path,
                device_output_path=self.device_output_response_path,
                callback=partial(
                    post_process_model_output,
                    output_holder=model_output_holder,
                    host_output_response_path=self.host_output_response_path,
                ),
            )
            self.adb.pull(
                host_output_path=self.host_performance_path,
                device_output_path=self.device_performance_path,
                callback=partial(
                    post_process_inference_speed,
                    output_holder=performance_holder,
                    host_performance_path=self.host_performance_path,
                ),
            )
        self.inference_speed = performance_holder[0]
        return model_output_holder


class SqnrEval(EvalBase):
    """
    SQNR Evaluator: compares FP32 nn.Module logits vs on-device QNN logits.

    """

    def __init__(
        self,
        source_model,
        get_example_inputs: Callable,
        args,
        pte_paths,
        tokenizer_wrapper: TokenizerWrapper,
        decoder_model_config: LLMModelConfig,
        runtime_tokenizer_path,
        is_multimodal,
        dataset_builder,
        encoder: Optional[Union[torch.nn.Module, torch.fx.GraphModule]] = None,
        tok_embedding: Optional[Union[torch.nn.Module, torch.fx.GraphModule]] = None,
        audio_token_id: Optional[int] = None,
        image_token_id: Optional[int] = None,
    ):
        super().__init__(
            args, pte_paths, runtime_tokenizer_path, is_multimodal, dataset_builder
        )
        self.inference_speed = 0
        self.source_model = source_model
        self.get_example_inputs = get_example_inputs
        self.adb = self._get_adb()
        self.tokenizer = tokenizer_wrapper.tokenizer
        self.enable_x86_64 = args.enable_x86_64
        self.max_seq_length = args.max_seq_len
        self.enable_attention_sink = args.use_attention_sink is not None

        pte_meta_info = retrieve_info_from_pte(pte_path=pte_paths[TEXT_DECODER])
        self.output_vocab_size = pte_meta_info["output_vocab_size"]
        pte_max_context_len = pte_meta_info["pte_max_context_len"]
        self.logits_scale = pte_meta_info["logits_scale"]
        self.logits_zero_point = pte_meta_info["logits_zero_point"]
        self.kv_io_bit_width = pte_meta_info["kv_io_bit_width"]

        self._inference = ModelInference(
            decoder=DecoderInference(
                get_example_inputs=get_example_inputs,
                audio_token_id=audio_token_id,
                image_token_id=image_token_id,
                max_context_len=self.max_seq_length,
                use_i64_token=args.embedding_quantize is not None,
            ),
            encoder=EncoderInference() if encoder is not None else None,
        )
        self._encoder_module = encoder
        self._tok_embedding_module = tok_embedding
        self.decoder_model_config = decoder_model_config

        if args.model_mode != "kv":
            logging.warning(
                f"Current Sqnr Eval does not support {args.model_mode}, switching to kv mode."
            )

        if pte_max_context_len != self.max_seq_length:
            logging.warning(
                f"The pte provided has a max_context_len {pte_max_context_len}, which is different from --max_seq_len {self.max_seq_length} provided to the script, please ensure this is desired."
            )
            if (
                not self.enable_attention_sink
                and pte_max_context_len < self.max_seq_length
            ):
                logging.warning(
                    f"The pte max_context_len {pte_max_context_len} is used since it is shorter than --max_seq_len {self.max_seq_length}"
                )
                self.max_seq_length = pte_max_context_len

    def run(self, message: MessageSample, audio_paths=None, image_paths=None):
        dataloaders = self.dataset_builder.build_runtime_dataloader(message)
        audio_dataloader = dataloaders.get(AUDIO_ENCODER)
        vision_dataloader = dataloaders.get(VISION_ENCODER)
        text_dataloader = dataloaders[TEXT_DECODER]
        audio_batch, vision_batch, text_batch = next(
            iter(
                zip(
                    safe_dataloader_iter(audio_dataloader),
                    safe_dataloader_iter(vision_dataloader),
                    text_dataloader,
                )
            )
        )
        token_ids = text_batch["token_ids"]
        input_ids, attn_mask = text_batch["input_ids"], text_batch["attention_mask"]
        encoder_inputs = (audio_batch or vision_batch or {}).get("inputs")
        logits = self._inference.predict_step(
            self.source_model,
            input_ids=input_ids,
            attn_mask=attn_mask,
            tok_embedding=self._tok_embedding_module,
            encoder_module=self._encoder_module,
            encoder_inputs=encoder_inputs,
        )

        input_file_name = f"{self.args.artifact}/input_tokens.raw"
        prompt_len = len(token_ids[0])
        golden_logits = logits[:, :prompt_len, :]
        inps = input_ids[0, :prompt_len].to(torch.uint64).numpy()
        inps.tofile(input_file_name)

        assert (
            inps.size < self.max_seq_length
        ), f"Number of input token is longer than max_seq_len, please shorten the input token length. input_token length: {inps.size}. max_seq_len: {self.max_seq_length}"

        output_logits_holder = []
        output_performance_holder = []

        runner_cmd = self.runner_base_cmd

        if self.is_multimodal:
            modality_inputs = preprocess_encoder_inputs(
                self.decoder_model_config,
                self.args.decoder_model,
                audio_paths or [],
                image_paths or [],
            )
            modality_input_cmd, self.modality_input_files = (
                self._init_multimodal_base_cmd(modality_inputs)
            )
            runner_cmd = " ".join(
                [
                    runner_cmd,
                    modality_input_cmd,
                ]
            )

        if self.enable_x86_64:
            runner_cmd = " ".join(
                [
                    runner_cmd,
                    f"--seq_len {inps.size + 1}",
                    f"--eval_mode {EVAL_MODE['kv']}",
                    "--temperature 0",
                    f"--dump_logits_path {self.device_logits_path}",
                    f"--tokenized_prompt {input_file_name}",
                ]
            )
            subprocess.run(
                runner_cmd,
                shell=True,
                executable="/bin/bash",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            post_process_logits(
                output_holder=output_logits_holder,
                host_logits_path=self.host_logits_path,
                kv_io_bit_width=self.kv_io_bit_width,
                num_input_tokens=inps.size,
                output_vocab_size=self.output_vocab_size,
                logits_scale=self.logits_scale,
                logits_zero_point=self.logits_zero_point,
            )
            post_process_inference_speed(
                output_holder=output_performance_holder,
                host_performance_path=self.host_performance_path,
            )
        else:
            runner_cmd = " ".join(
                [
                    runner_cmd,
                    f"--seq_len {inps.size + 1}",
                    f"--eval_mode {EVAL_MODE['kv']}",
                    "--temperature 0",
                    f"--dump_logits_path {self.device_logits_path}",
                    f"--tokenized_prompt {os.path.basename(input_file_name)}",
                ]
            )
            extra_files = [input_file_name, self.runtime_tokenizer_path]
            if self.is_multimodal:
                extra_files.extend(self.modality_input_files)
            self.adb.push(
                inputs=[],
                files=extra_files,
            )
            self.adb.execute(custom_runner_cmd=runner_cmd)
            self.adb.pull(
                host_output_path=self.host_logits_path,
                device_output_path=self.device_logits_path,
                callback=partial(
                    post_process_logits,
                    output_holder=output_logits_holder,
                    host_logits_path=self.host_logits_path,
                    kv_io_bit_width=self.kv_io_bit_width,
                    num_input_tokens=inps.size,
                    output_vocab_size=self.output_vocab_size,
                    logits_scale=self.logits_scale,
                    logits_zero_point=self.logits_zero_point,
                ),
            )
            self.adb.pull(
                host_output_path=self.host_performance_path,
                device_output_path=self.device_performance_path,
                callback=partial(
                    post_process_inference_speed,
                    output_holder=output_performance_holder,
                    host_performance_path=self.host_performance_path,
                ),
            )
        self.inference_speed = output_performance_holder[0]

        sqnr = compute_error(golden_logits, output_logits_holder[0])
        return sqnr.item(), golden_logits, output_logits_holder[0]



def _device_temps(serial):
    """(SoC °C, 배터리 °C) 를 읽는다. 못 읽으면 해당 값은 None."""
    def sh(cmd):
        try:
            r = subprocess.run(["adb", "-s", serial, "shell", cmd],
                               capture_output=True, text=True, timeout=25)
            return r.stdout.strip() if r.returncode == 0 else ""
        except Exception:
            return ""
    soc = None
    raw = sh("cat /sys/class/thermal/thermal_zone0/temp")
    for tok in raw.split():
        if tok.lstrip("-").isdigit():
            soc = int(tok) / 1000.0
            break
    batt = None
    raw = sh("dumpsys battery | grep -m1 '  temperature:'")
    for tok in raw.replace("temperature:", " ").split():
        if tok.lstrip("-").isdigit():
            batt = int(tok) / 10.0
            break
    lvl = None
    raw = sh("dumpsys battery | grep -m1 '  level:'")
    for tok in raw.replace("level:", " ").split():
        if tok.isdigit():
            lvl = int(tok)
            break
    return soc, batt, lvl


def thermal_gate(serial):
    """기기가 충분히 식을 때까지 기다린다.

    2026-09-09: 냉각 없이 8문서를 연속으로 돌리다 폰이 꺼졌다. 당시 로그를 보면
    SoC 65.7 도, 배터리 52.7 도였고 USB 연결 상태인데도 -2.4 A 로 방전 중이었다
    (NPU 부하가 USB 공급량을 넘김). 그래서 창(window) 하나마다 여기서 막는다.

    env:
      QNN_COOL_SOC_C   재개 허용 SoC 온도 (기본 42)
      QNN_COOL_BATT_C  재개 허용 배터리 온도 (기본 40)
      QNN_COOL_MIN_PCT 이 배터리 잔량 미만이면 중단 (기본 25)
      QNN_COOL_MAX_S   최대 대기 초 (기본 900). 넘으면 경고 후 진행
      QNN_COOL_OFF=1   게이트 비활성화
    """
    import time as _t
    if os.environ.get("QNN_COOL_OFF") == "1":
        return
    soc_max = float(os.environ.get("QNN_COOL_SOC_C", "42"))
    batt_max = float(os.environ.get("QNN_COOL_BATT_C", "40"))
    min_pct = int(os.environ.get("QNN_COOL_MIN_PCT", "25"))
    max_wait = float(os.environ.get("QNN_COOL_MAX_S", "900"))
    t0 = _t.time()
    waited = False
    while True:
        soc, batt, lvl = _device_temps(serial)
        if lvl is not None and lvl < min_pct:
            raise RuntimeError(
                f"battery {lvl}% < {min_pct}%: 기기를 충전한 뒤 다시 실행하세요"
            )
        hot = (soc is not None and soc > soc_max) or (
            batt is not None and batt > batt_max
        )
        if not hot:
            if waited:
                logging.info(
                    "[THERMAL] 재개 soc=%s batt=%s lvl=%s%% (대기 %.0f초)",
                    soc, batt, lvl, _t.time() - t0,
                )
            else:
                logging.info("[THERMAL] soc=%s batt=%s lvl=%s%% 진행", soc, batt, lvl)
            return
        if _t.time() - t0 > max_wait:
            logging.warning(
                "[THERMAL] %.0f초 대기했으나 soc=%s batt=%s 로 여전히 뜨겁다. 진행한다.",
                max_wait, soc, batt,
            )
            return
        if not waited:
            logging.info(
                "[THERMAL] 냉각 대기 시작 soc=%s(>%.0f) batt=%s(>%.0f) lvl=%s%%",
                soc, soc_max, batt, batt_max, lvl,
            )
            waited = True
        _t.sleep(20)


class TaskEval(EvalBase):
    class QnnRunnerEvalWrapper(EagerEvalWrapper):
        """Runs lm_eval tasks on device via QNN runner."""

        def __init__(  # noqa: C901
            self,
            args,
            runner_base_cmd: str,
            adb: SimpleADB,
            pte_path: str,
            device_performance_path: str,
            device_logits_path: str,
            host_performance_path: str,
            host_logits_path: str,
            tokenizer: Union[
                SentencePieceTokenizer, TiktokenTokenizer, HuggingFaceTokenizer
            ],
            runtime_tokenizer_path,
        ):
            self.inference_speed = None

            self.args = args
            self.runner_base_cmd = runner_base_cmd
            self.adb = adb
            self.pte_path = pte_path
            self.runtime_tokenizer_path = runtime_tokenizer_path

            self.device_performance_path = device_performance_path
            self.device_logits_path = device_logits_path
            self.host_performance_path = host_performance_path
            self.host_logits_path = host_logits_path

            self.enable_x86_64 = args.enable_x86_64
            self.max_seq_length = args.max_seq_len
            self.enable_attention_sink = args.use_attention_sink is not None
            pte_meta_info = retrieve_info_from_pte(pte_path=self.pte_path)
            self.output_vocab_size = pte_meta_info["output_vocab_size"]
            pte_max_context_len = pte_meta_info["pte_max_context_len"]
            self.logits_scale = pte_meta_info["logits_scale"]
            self.logits_zero_point = pte_meta_info["logits_zero_point"]
            self.kv_io_bit_width = pte_meta_info["kv_io_bit_width"]
            # 축약 덤프(compact): 윈도우마다 545 MB logits 를 pull 하지 않고, 러너가 행마다
            # logsumexp/argmax/정답 logit 만 기기에 남겨 두면 배치 끝에 한 번만 가져온다.
            # 2026-09-09: USB 끊김 28 건이 전부 그 pull 도중에 났다. QNN_PPL_COMPACT=0 으로 끈다.
            self.compact = (
                os.environ.get("QNN_PPL_COMPACT", "1") == "1" and not self.enable_x86_64
            )
            self.device_compact_dir = f"{os.path.dirname(device_logits_path)}/compact"
            self.host_compact_dir = f"{os.path.dirname(host_logits_path)}/compact"

            if args.model_mode != "kv":
                logging.warning(
                    f"Current QnnRunnerEvalWrapper does not support {args.model_mode}, switching to kv mode."
                )

            if pte_max_context_len != self.max_seq_length:
                logging.warning(
                    f"The pte provided has a max_context_len {pte_max_context_len}, which is different from --max_seq_len {self.max_seq_length} provided to the script, please ensure this is desired."
                )
                if (
                    not self.enable_attention_sink
                    and pte_max_context_len < self.max_seq_length
                ):
                    logging.warning(
                        f"The pte max_context_len {pte_max_context_len} is used since it is shorter than --max_seq_len {self.max_seq_length}"
                    )
                    self.max_seq_length = pte_max_context_len

            if not self.enable_x86_64:
                self.adb.push(inputs=[], files=[self.runtime_tokenizer_path])
            # pyre-ignore
            super().__init__(None, tokenizer, self.max_seq_length - 1)

        @property
        def batch_size(self):
            # compact 모드: lm_eval 이 모든 윈도우를 한 번의 _loglikelihood_tokens 호출로
            # 넘기게 해서, 기기 실행을 전부 마친 뒤 결과를 한 번에 pull 한다.
            return 10**6 if self.compact else 1

        _COMPACT_ROW = np.dtype(
            [("lse", "<f4"), ("target_logit", "<f4"), ("argmax", "<i4"), ("target", "<i4")]
        )

        def _read_compact(self, path):
            with open(path, "rb") as f:
                head = f.read(12)
                assert head[:4] == b"CLG1", f"bad compact logits file: {path}"
                n_rows, _vocab = np.frombuffer(head[4:], dtype="<i4")
                rows = np.frombuffer(f.read(), dtype=self._COMPACT_ROW, count=int(n_rows))
            return rows

        def _loglikelihood_tokens(self, requests, disable_tqdm=False, override_bs=None):
            if not self.compact:
                return super()._loglikelihood_tokens(
                    requests, disable_tqdm=disable_tqdm, override_bs=override_bs
                )
            serial = getattr(self.adb, "device_id", None) or os.environ.get(
                "QNN_DEVICE_SERIAL", ""
            )
            scale = 1.0 if self.logits_scale is None else float(self.logits_scale)
            zp = 0 if self.logits_zero_point is None else int(self.logits_zero_point)
            # QNN_PPL_RESUME=1: 이미 끝난 창의 결과를 기기에 남겨 두고 건너뛴다.
            # 끊기거나 중단돼도 처음부터 다시 안 돌리려는 것 (2026-09-09).
            resume = os.environ.get("QNN_PPL_RESUME", "1") == "1"
            if resume:
                self.adb._adb(["shell", f"mkdir -p {self.device_compact_dir}"])
                done = set(
                    subprocess.run(
                        self.adb._adb_prefix()
                        + ["shell", f"ls {self.device_compact_dir} 2>/dev/null"],
                        capture_output=True,
                        text=True,
                    ).stdout.split()
                )
            else:
                self.adb._adb(
                    ["shell", f"rm -rf {self.device_compact_dir} && mkdir -p {self.device_compact_dir}"]
                )
                done = set()
            windows = []
            n = len(requests)
            for i, (cache_key, context_enc, continuation_enc) in enumerate(requests):
                # lm_eval HFLM._loglikelihood_tokens 와 같은 창 구성: 입력은 마지막 토큰을 뺀
                # 것, 정답은 한 칸 민 것. 연속(continuation) 구간은 마지막 contlen 개 위치.
                full = (context_enc + continuation_enc)[-(self.max_length + 1) :]
                inp = np.asarray(full[:-1], dtype=np.uint64)
                tgt = np.asarray(full[1:], dtype=np.uint64)
                input_file = f"{self.args.artifact}/input_tokens.raw"
                target_file = f"{self.args.artifact}/target_tokens.raw"
                inp.tofile(input_file)
                tgt.tofile(target_file)
                # 창 번호만으로 재사용하면 limit 이 바뀌었을 때 내용이 다른 창을 건너뛴다.
                # 입력 토큰 해시를 파일명에 넣어 같은 내용일 때만 이어받는다.
                tag = hashlib.md5(inp.tobytes()).hexdigest()[:8]
                dev_out = f"{self.device_compact_dir}/w{i:05d}_{tag}.bin"
                dev_perf = f"{self.device_compact_dir}/perf{i:05d}_{tag}.txt"
                runner_cmd = " ".join(
                    [
                        self.runner_base_cmd,
                        f"--seq_len {self.max_seq_length}",
                        f"--eval_mode {EVAL_MODE['kv']}",
                        "--temperature 0",
                        f"--dump_logits_path {dev_out}",
                        "--dump_logits_compact",
                        f"--dump_targets_path {os.path.basename(target_file)}",
                        f"--logits_scale {scale!r}",
                        f"--logits_zero_point {zp}",
                        f"--performance_output_path {dev_perf}",
                        f"--tokenized_prompt {os.path.basename(input_file)}",
                    ]
                )
                if os.path.basename(dev_out) in done and os.path.basename(dev_perf) in done:
                    logging.info(f"[compact] window {i + 1}/{n}: 이미 완료, 건너뜀")
                else:
                    self.adb.push(
                        inputs=[], files=[input_file, target_file], init_env=False
                    )
                    thermal_gate(serial)
                    logging.info(f"[compact] window {i + 1}/{n}: {inp.size} tokens")
                    self.adb.execute(custom_runner_cmd=runner_cmd)
                windows.append(
                    (cache_key, len(continuation_enc), int(inp.size), tag)
                )
            # 결과를 한 번에 pull (윈도우당 ~14 KB)
            shutil.rmtree(self.host_compact_dir, ignore_errors=True)
            self.adb.pull(
                host_output_path=os.path.dirname(self.host_compact_dir),
                device_output_path=self.device_compact_dir,
            )
            res = []
            speeds = []
            for i, (cache_key, contlen, inplen, tag) in enumerate(windows):
                rows = self._read_compact(f"{self.host_compact_dir}/w{i:05d}_{tag}.bin")
                assert rows.shape[0] >= inplen, (
                    f"window {i}: {rows.shape[0]} rows < {inplen} input tokens"
                )
                sel = rows[inplen - contlen : inplen]
                ll = float(
                    np.sum(sel["target_logit"].astype(np.float64) - sel["lse"].astype(np.float64))
                )
                greedy = bool(np.all(sel["argmax"] == sel["target"]))
                res.append((ll, greedy))
                if cache_key is not None:
                    self.cache_hook.add_partial("loglikelihood", cache_key, (ll, greedy))
                try:
                    with open(f"{self.host_compact_dir}/perf{i:05d}_{tag}.txt") as f:
                        speeds.append(float(f.read()))
                except (OSError, ValueError):
                    pass
            if speeds:
                self.inference_speed = speeds[-1]
            return res

        def _model_call(self, inps):
            _, seq_len = inps.shape
            input_file_name = f"{self.args.artifact}/input_tokens.raw"
            inps = inps.to(torch.uint64).numpy()
            inps.tofile(input_file_name)
            output_logits_holder = []
            output_performance_holder = []

            if self.enable_x86_64:
                runner_cmd = " ".join(
                    [
                        self.runner_base_cmd,
                        f"--seq_len {self.max_seq_length}",
                        f"--eval_mode {EVAL_MODE['kv']}",
                        "--temperature 0",
                        f"--dump_logits_path {self.device_logits_path}",
                        f"--tokenized_prompt {input_file_name}",
                    ]
                )
                subprocess.run(
                    runner_cmd,
                    shell=True,
                    executable="/bin/bash",
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                post_process_logits(
                    output_holder=output_logits_holder,
                    host_logits_path=self.host_logits_path,
                    kv_io_bit_width=self.kv_io_bit_width,
                    num_input_tokens=inps.size,
                    output_vocab_size=self.output_vocab_size,
                    logits_scale=self.logits_scale,
                    logits_zero_point=self.logits_zero_point,
                )
                post_process_inference_speed(
                    output_holder=output_performance_holder,
                    host_performance_path=self.host_performance_path,
                )
            else:
                runner_cmd = " ".join(
                    [
                        self.runner_base_cmd,
                        f"--seq_len {self.max_seq_length}",
                        f"--eval_mode {EVAL_MODE['kv']}",
                        "--temperature 0",
                        f"--dump_logits_path {self.device_logits_path}",
                        f"--tokenized_prompt {os.path.basename(input_file_name)}",
                    ]
                )
                self.adb.push(inputs=[], files=[input_file_name], init_env=False)
                # 창 하나마다 발열 게이트 (위 thermal_gate 주석 참고)
                thermal_gate(getattr(self.adb, "device_id", None)
                             or os.environ.get("QNN_DEVICE_SERIAL", ""))
                self.adb.execute(custom_runner_cmd=runner_cmd)
                self.adb.pull(
                    host_output_path=self.host_logits_path,
                    device_output_path=self.device_logits_path,
                    callback=partial(
                        post_process_logits,
                        output_holder=output_logits_holder,
                        host_logits_path=self.host_logits_path,
                        kv_io_bit_width=self.kv_io_bit_width,
                        num_input_tokens=inps.size,
                        output_vocab_size=self.output_vocab_size,
                        logits_scale=self.logits_scale,
                        logits_zero_point=self.logits_zero_point,
                    ),
                )
                self.adb.pull(
                    host_output_path=self.host_performance_path,
                    device_output_path=self.device_performance_path,
                    callback=partial(
                        post_process_inference_speed,
                        output_holder=output_performance_holder,
                        host_performance_path=self.host_performance_path,
                    ),
                )
            self.inference_speed = output_performance_holder[0]
            return output_logits_holder[0][:, :seq_len, :]

    def __init__(
        self, args, pte_paths, tokenizer, runtime_tokenizer_path, is_multimodal
    ):
        super().__init__(
            args=args,
            pte_paths=pte_paths,
            runtime_tokenizer_path=runtime_tokenizer_path,
            is_multimodal=is_multimodal,
        )
        self.inference_speed = None
        self.tasks = args.eval_tasks
        self.num_fewshot = args.eval_num_fewshot
        self.limit = args.eval_limit
        adb = self._get_adb()
        self.eval_wrapper = TaskEval.QnnRunnerEvalWrapper(
            args=args,
            runner_base_cmd=self.runner_base_cmd,
            adb=adb,
            pte_path=self.pte_paths[TEXT_DECODER],
            device_performance_path=self.device_performance_path,
            device_logits_path=self.device_logits_path,
            host_performance_path=self.host_performance_path,
            host_logits_path=self.host_logits_path,
            tokenizer=tokenizer,
            runtime_tokenizer_path=self.runtime_tokenizer_path,
        )

    def run(self):
        with torch.no_grad():
            eval_results = simple_evaluate(
                model=self.eval_wrapper,
                tasks=self.tasks,
                num_fewshot=self.num_fewshot,
                limit=self.limit,
            )
        self.inference_speed = self.eval_wrapper.inference_speed
        return eval_results
