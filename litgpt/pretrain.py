# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

from contextlib import nullcontext
import math
import pprint
import time
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Optional, Tuple, Union, Any
import os
import sys

import lightning as L
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from lightning.fabric.strategies import FSDPStrategy
from lightning.fabric.utilities.throughput import ThroughputMonitor, measure_flops
import torch.profiler as tprofiler
from torch.utils.data import DataLoader
from torchmetrics.aggregation import RunningMean
from typing_extensions import Literal
import nvtx

from litgpt import Tokenizer
from litgpt.args import EvalArgs, TrainArgs
from litgpt.config import name_to_config
from litgpt.data import DataModule, RedRockOpenWebText
from litgpt.model import GPT, Block, CausalSelfAttention, Config, LLaMAMLP
from litgpt.utils import (
    CLI,
    CycleIterator,
    capture_hparams,
    choose_logger,
    chunked_cross_entropy,
    copy_config_files,
    get_default_supported_precision,
    init_out_dir,
    num_parameters,
    parse_devices,
    reset_parameters,
    save_config,
    save_hyperparameters,
)

mp.set_start_method("spawn", force=True)
try:
    import utilities.monitor_collectives

    utilities.monitor_collectives.shunt_torch_communication()
except ModuleNotFoundError as e:
    print(e)
    print("Monitor collectives library not found. Collectives will not be monitored..")


def setup(
    model_name: Optional[str] = None,
    model_config: Optional[Config] = None,
    out_dir: Path = Path("out/pretrain"),
    precision: Literal["bf16-true", "bf16-mixed", "32-true", None] = None,
    initial_checkpoint_dir: Optional[Path] = None,
    resume: Union[bool, Path] = False,
    data_dir: Path = Path("data"),
    num_nodes: int = 1,
    use_pt_profiler: bool = False,
    train: TrainArgs = TrainArgs(
        save_interval=int(os.environ.get("SAVE_INTERVAL", 10000)),
        log_interval=1,
        global_batch_size=512,
        micro_batch_size=4,
        max_tokens=int(3e12),  # 3 trillion
        max_steps=None,
        max_seq_length=None,
        learning_rate=4e-4,
        weight_decay=1e-1,
        beta1=0.9,
        beta2=0.95,
        max_norm=1.0,
        min_lr=4e-5,
        lr_warmup_steps=2000,
        tie_embeddings=False,
        fast_init=True,
    ),
    eval: EvalArgs = EvalArgs(interval=1000, max_iters=100),
    devices: Union[int, str] = "auto",
    tokenizer_dir: Optional[Path] = None,
    logger_name: Literal["wandb", "tensorboard", "csv"] = "csv",
    seed: int = 42,
    replication_size: int = None,
    sharding_size: int = None,
):
    """Pretrain a model.

    Arguments:
        model_name: The name of the model to pretrain. Choose from names in ``litgpt.config``. Mutually exclusive with
            ``model_config``.
        model_config: A ``litgpt.Config`` object to define the model architecture. Mutually exclusive with
            ``model_config``.
        out_dir: Directory in which to save checkpoints and logs. If running in a Lightning Studio Job, look for it in
            /teamspace/jobs/<job-name>/share.
        precision: The precision to use for finetuning. Determines a compatible precision setting by default.
        initial_checkpoint_dir: Optional path to a checkpoint directory to initialize the model from.
            Useful for continued pretraining. Mutually exclusive with ``resume``.
        resume: Path to a checkpoint directory to resume from in case training was interrupted, or ``True`` to resume
            from the latest checkpoint in ``out_dir``.
        data_dir: Directory in which train and val data files are located.
        num_nodes: The number of nodes to train on.
        use_pt_profiler: Whether to use torch.profiler or nvtx markers. 
        train: Training-related arguments. See ``litgpt.args.TrainArgs`` for details.
        eval: Evaluation-related arguments. See ``litgpt.args.EvalArgs`` for details.
        devices: How many devices/GPUs to use. Uses all GPUs by default.
        tokenizer_dir: Optional path to the tokenizer dir that was used for preprocessing the dataset. Only some data
            module require this.
        logger_name: The name of the logger to send metrics to.
        seed: The random seed to use for reproducibility.
        replication_size: The number of model replicas. Only relevant when using FSDP.
        sharding_size: The number of devices to shard per FSDP group. Only relevant when using FSDP.
    """
    hparams = capture_hparams()
    data = RedRockOpenWebText(data_path=data_dir, num_workers=2, batch_size=train.micro_batch_size)
    if model_config is not None and model_name is not None:
        raise ValueError("Only one of `model_name` or `model_config` can be set.")
    elif model_config is None and model_name is None:
        available_models = "\n".join(sorted(name_to_config))
        raise ValueError(f"Please specify --model_name <model_name>. Available values:\n{available_models}")
    config = Config.from_name(model_name) if model_config is None else model_config
    precision = precision or get_default_supported_precision(training=True)
    devices = parse_devices(devices)
    out_dir = init_out_dir(out_dir)
    # in case the dataset requires the Tokenizer
    tokenizer = Tokenizer(tokenizer_dir) if tokenizer_dir is not None else None
 
    logger = choose_logger(
        logger_name, out_dir, name=f"pretrain-{config.name}", resume=resume, log_interval=train.log_interval
    )

    if devices > 1:
        if isinstance(replication_size, int) ^ isinstance(sharding_size, int):
            raise ValueError("Either `replication_size` and `sharding_size` must both be set or left undefined")
        device_mesh = (replication_size, sharding_size) if replication_size else None
        strategy = FSDPStrategy(auto_wrap_policy={Block}, state_dict_type="full", sharding_strategy="HYBRID_SHARD", device_mesh=device_mesh)
    else:
        strategy = "auto"
    fabric = L.Fabric(devices=devices, num_nodes=num_nodes, strategy=strategy, precision=precision, loggers=[logger])
    fabric.launch()

    fabric.print(pprint.pformat(hparams))
    if logger_name in ("tensorboard", "wandb"):
        fabric.logger.log_hyperparams(hparams)

    # Configure profiler
    pt_profiler_wait=1
    pt_profiler_warmup=2
    pt_profiler_active=2
    pt_profiler_repeat=5
    nsys_profile_step_multiple=5

    main(
        fabric,
        devices,
        seed,
        initial_checkpoint_dir,
        resume,
        config,
        data,
        out_dir,
        tokenizer_dir,
        tokenizer,
        use_pt_profiler,
        pt_profiler_wait,
        pt_profiler_warmup,
        pt_profiler_active,
        pt_profiler_repeat,
        nsys_profile_step_multiple,
        train,
        eval,
    )


def main(
    fabric: L.Fabric,
    devices: int,
    seed: int,
    initial_checkpoint_dir: Optional[Path],
    resume: Union[bool, Path],
    config: Config,
    data: DataModule,
    out_dir: Path,
    tokenizer_dir: Optional[Path],
    tokenizer: Optional[Tokenizer],
    use_pt_profiler: bool,
    pt_profiler_wait: int,
    pt_profiler_warmup: int,
    pt_profiler_active: int,
    pt_profiler_repeat: int,
    nsys_profile_step_multiple: int,
    train: TrainArgs,
    eval: EvalArgs,
) -> None:
    if use_pt_profiler:
        cm = nullcontext()
    else:
        cm = torch.autograd.profiler.emit_nvtx()
    with cm:
        validate_args(train, eval, initial_checkpoint_dir, resume)

        if fabric.global_rank == 0:
            out_dir.mkdir(parents=True, exist_ok=True)
        tprofiler_out_dir = out_dir / "tprofiler"
        execution_trace_out_dir = out_dir / "execution_trace"
        execution_trace_out_dir.mkdir(parents=True, exist_ok=True)

        fabric.seed_everything(seed)  # same seed for every process to init model (FSDP)

        if use_pt_profiler:
            prof = tprofiler.profile(
                schedule=tprofiler.schedule(
                    wait=pt_profiler_wait,
                    warmup=pt_profiler_warmup,
                    active=pt_profiler_active,
                    repeat=pt_profiler_repeat,
                ),
                on_trace_ready=tprofiler.tensorboard_trace_handler(tprofiler_out_dir),
                record_shapes=True,
                with_stack=True,
                activities=[tprofiler.ProfilerActivity.CPU, tprofiler.ProfilerActivity.CUDA],
            )
            prof.start()
        else:
            prof = None


        t0 = time.perf_counter()
        with fabric.init_module(empty_init=True):
            model = GPT(config)

        initialize_weights(fabric, model, n_layer=config.n_layer, n_embd=config.n_embd, fast_init=train.fast_init)

        if train.tie_embeddings:
            model.transformer.wte.weight = model.lm_head.weight
        if train.max_seq_length:
            model.max_seq_length = train.max_seq_length

        fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
        fabric.print(f"Total parameters: {num_parameters(model):,}")

        model = torch.compile(model)
        model = fabric.setup(model)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train.learning_rate,
            weight_decay=train.weight_decay,
            betas=(train.beta1, train.beta2),
            fused=fabric.device.type == "cuda",
        )
        optimizer = fabric.setup_optimizers(optimizer)

        train_dataloader, val_dataloader = get_dataloaders(fabric, data, tokenizer, train, model.max_seq_length)
        train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

        if initial_checkpoint_dir:
            fabric.load_raw(initial_checkpoint_dir / "lit_model.pth", model)

        state = {
            "model": model,
            "optimizer": optimizer,
            "train_dataloader": train_dataloader,
            "iter_num": 0,
            "step_count": 0,
        }

        if resume is True:
            resume = max(out_dir.rglob("step-*/*.pth"), key=(lambda p: int(p.parent.name.split("-")[1])))
        if resume:
            fabric.print(f"Resuming training from {resume}")
            fabric.load(resume, state)

        train_time = time.perf_counter()
        fit(fabric, devices, state, train_dataloader, val_dataloader, out_dir, tokenizer_dir, train, eval, prof, nsys_profile_step_multiple)

        if prof:
            prof.stop()

        fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
        if fabric.device.type == "cuda":
            fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


def fit(
    fabric: L.Fabric,
    devices: int,
    state: dict,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    out_dir: Path,
    tokenizer_dir: Optional[Path],
    train: TrainArgs,
    eval: EvalArgs,
    prof: tprofiler.profile,
    nsys_profile_step_multiple: int = 5,
) -> None:
    model = state["model"]
    optimizer = state["optimizer"]

    if eval.initial_validation:
        val_loss = validate(fabric, model, val_dataloader, max_iters=eval.max_iters)
        val_loss = f"{val_loss:.3f}"
    else:
        validate(fabric, model, val_dataloader, max_iters=2)   # sanity check
        val_loss = "n/a"

    throughput = ThroughputMonitor(fabric, window_size=5)

    with torch.device("meta"):
        meta_model = GPT(model.config)
        x = torch.randint(0, 1, (train.micro_batch_size, meta_model.max_seq_length))
        model_fwd = lambda: meta_model(x)
        model_loss = lambda y: chunked_cross_entropy(y, x, chunk_size=0)
        measured_flops = measure_flops(meta_model, model_fwd, model_loss)
        fabric.print(f"Measured TFLOPs: {measured_flops * fabric.world_size / 1e12:.2f}")
        del meta_model, x

    max_tokens_per_device = train.max_tokens // fabric.world_size
    tokens_per_iter = train.micro_batch_size * model.max_seq_length
    max_iters = max_tokens_per_device // tokens_per_iter
    log_iter_interval = train.log_interval * train.gradient_accumulation_iters(devices)
    initial_iter = state["iter_num"]
    train_iterator = CycleIterator(train_dataloader)

    running_loss = RunningMean(window=train.gradient_accumulation_iters(devices), sync_on_compute=False).to(
        fabric.device
    )
    fabric.barrier()
    total_t0 = time.perf_counter()

    warmup_iters = train.warmup_iters(devices, max_iters, train_dataloader)

    for train_data in train_iterator:
        if state["iter_num"] >= max_iters:
            break

        if state["step_count"] >= train.max_steps:
            break

        # determine and set the learning rate for this iteration
        lr = get_lr(train.learning_rate, state["iter_num"], warmup_iters, max_iters, train.min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        state["iter_num"] += 1

        # is_accumulating is True for all but the last microbatch in each step
        is_accumulating = state["iter_num"] % train.gradient_accumulation_iters(devices) != 0
        capture_profile = state["step_count"] > 0 and state["step_count"] % nsys_profile_step_multiple == 0

        if capture_profile and (state["iter_num"] - 1) % train.gradient_accumulation_iters(devices) == 0:
            fabric.print(f"Starting Nsys profiling.")
            torch.cuda.profiler.start()

        iter_t0 = time.perf_counter()
        input_ids, targets = train_data

        with nvtx.annotate(color="green", message=f"training_step"):
            with fabric.no_backward_sync(model, enabled=is_accumulating):
                with nvtx.annotate(color="blue", message=f"forward_step"):
                    logits = model(input_ids)
                    loss = chunked_cross_entropy(logits, targets)
                
                with nvtx.annotate(color="red", message=f"backward_step"):
                    fabric.backward(loss / train.gradient_accumulation_iters(devices))

            running_loss.update(loss.detach())

            if not is_accumulating:
                fabric.clip_gradients(model, optimizer, max_norm=train.max_norm)
                with nvtx.annotate(color="yellow", message=f"optimizer_step"):
                    optimizer.step()
                
                optimizer.zero_grad()
                state["step_count"] += 1

                if prof:
                    prof.step()

        if state["iter_num"] % log_iter_interval == 0:
            with nvtx.annotate(color="purple", message="device_to_host_logging_sync"):
                loss = running_loss.compute().item()  # expensive device-to-host synchronization

            t1 = time.perf_counter()
            throughput.update(
                time=(t1 - total_t0),
                flops=(measured_flops * log_iter_interval),
                batches=state["iter_num"],
                samples=(state["iter_num"] * train.micro_batch_size),
                lengths=(state["iter_num"] * train.micro_batch_size * model.max_seq_length),
            )
            metrics = {
                "loss": loss,
                "iter": state["iter_num"],
                "step": state["step_count"],
                "epoch": train_iterator.epoch,
                "iter_time": t1 - iter_t0,
                "remaining_time": (
                    (t1 - total_t0) / (state["iter_num"] - initial_iter) * (max_iters - state["iter_num"])
                ),
                "tokens": state["iter_num"] * train.micro_batch_size * model.max_seq_length,
                "total_tokens": (state["iter_num"] * train.micro_batch_size * model.max_seq_length * fabric.world_size),
                "learning_rate": lr,
            }
            if isinstance(val_loss, float):
                val_loss = f"{val_loss:.3f}"
            fabric.print(
                f"Epoch {metrics['epoch']+1} | iter {metrics['iter']} step {metrics['step']} |"
                f" loss train: {metrics['loss']:.3f},"
                f" val: {val_loss} |"
                f" iter time: {metrics['iter_time'] * 1000:.2f} ms"
                f"{' (step)' if not is_accumulating else ''}"
                f" remaining time: {timedelta(seconds=int(metrics['remaining_time']))!s}"
            )

            fabric.print(f"HEARTBEAT: Step {state['step_count']}")

            throughput_metrics = throughput.compute()
            metrics.update(throughput_metrics)
            fabric.log_dict(metrics, step=state["iter_num"] - 1)

        if not is_accumulating and capture_profile:
            fabric.print(f"Stopping Nsys profiling.")
            torch.cuda.profiler.stop()

        if val_dataloader is not None and not is_accumulating and state["step_count"] % eval.interval == 0:
            t0 = time.perf_counter()
            val_loss = validate(fabric, model, val_dataloader, max_iters=eval.max_iters)
            val_loss = val_loss.item()
            td = time.perf_counter() - t0

            fabric.print(f"iter {state['iter_num']}: val loss {val_loss:.4f}, val time: {td * 1000:.2f} ms")
            metrics = {"val_loss": val_loss, "val_ppl": math.exp(val_loss)}
            fabric.log_dict(metrics, step=state["iter_num"] - 1)
            fabric.barrier()

        if train.save_interval is not None and not is_accumulating and state["step_count"] % train.save_interval == 0:
            save_checkpoint(fabric, state, tokenizer_dir, out_dir / f"step-{state['step_count']:08d}" / "lit_model.pth")


@torch.no_grad()
def validate(fabric: L.Fabric, model: nn.Module, val_dataloader: DataLoader, max_iters: int) -> torch.Tensor:
    fabric.barrier()
    fabric.print("Validating ...")
    model.eval()

    losses = []
    for k, batch in enumerate(val_dataloader):
        if k >= max_iters:
            break
        input_ids, targets = batch
        logits = model(input_ids)
        loss = chunked_cross_entropy(logits, targets)
        losses.append(loss)

    val_loss = torch.stack(losses).mean()
    model.train()
    fabric.barrier()
    return val_loss


def get_dataloaders(
    fabric: L.Fabric, data: DataModule, tokenizer: Tokenizer, train: TrainArgs, block_size: int
) -> Tuple[DataLoader, DataLoader]:
    data.connect(tokenizer=tokenizer, batch_size=train.micro_batch_size, max_seq_length=block_size)
    with fabric.rank_zero_first():
        data.prepare_data()
    data.setup()
    train_dataloader = data.train_dataloader()
    val_dataloader = data.val_dataloader()
    return train_dataloader, val_dataloader


# learning rate decay scheduler (cosine with linear warmup)
def get_lr(learning_rate: float, it: int, warmup_iters: int, max_iters: int, min_lr: float) -> float:
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > max_iters, return min learning rate
    if it > max_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


def initialize_weights(fabric: L.Fabric, model: GPT, n_layer: int, n_embd: int, fast_init: bool) -> None:
    """GPT-NeoX weight initialization (https://arxiv.org/abs/2204.06745)."""
    # Adapted from https://github.com/jzhang38/TinyLlama

    def init_weights(module, std):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)
    
    fabric.print(f"Initializing weights with {fast_init=}.", flush=True)
    if fast_init:
        t = time.time()
        for name, module in model.named_modules():
            state_dict = {}

            if isinstance(module, (nn.Linear, LLaMAMLP, CausalSelfAttention)):
                # define new layer on cuda so weight initialization is much faster
                in_features, out_features, bias, dtype = None, None, None, None
                if isinstance(module, (LLaMAMLP, CausalSelfAttention)):
                    in_features = module.proj.in_features
                    out_features = module.proj.out_features
                    bias = True if module.proj.bias is not None else False
                    dtype = module.proj.weight.dtype
                else:
                    in_features = module.in_features
                    out_features = module.out_features
                    bias = True if module.bias is not None else False
                    dtype = module.weight.dtype

                with torch.device("cuda"):
                    new_linear = torch.nn.Linear(
                        in_features,
                        out_features,
                        bias=bias,
                        dtype=dtype
                    )

                # initialize weights
                new_linear.apply(model._init_weights)

                # move new layer to cpu & prepare to load into model
                new_linear.to("cpu")
                
                if isinstance(module, (LLaMAMLP, CausalSelfAttention)):
                    state_dict[f"{name}.proj.weight"] = new_linear.weight
                    if module.proj.bias is not None:
                        state_dict[f"{name}.proj.bias"] = new_linear.bias
                else:
                    state_dict[f"{name}.weight"] = new_linear.weight
                    if module.bias is not None:
                        state_dict[f"{name}.bias"] = new_linear.bias

            elif isinstance(module, nn.Embedding):
                # define new layer on cuda so weight initialization is much faster
                with torch.device("cuda"):
                    new_embedding = torch.nn.Embedding(
                        module.weight.size()[0],
                        module.weight.size()[1],
                        dtype=module.weight.dtype
                    )

                # initialize weights
                new_embedding.apply(model._init_weights)

                # move new layer to cpu & prepare to load into model
                new_embedding.to("cpu")
                state_dict[f"{name}.weight"] = new_embedding.weight
 
            # load new layer's weights & biases into model
            model.load_state_dict(state_dict, strict=False, assign=True)
    else:
        t = time.time()
        for mod in model.modules():
            if isinstance(mod, (nn.Embedding, nn.Linear)):
                mod.reset_parameters = partial(init_weights, mod, std=math.sqrt(2.0 / 5 / n_embd))

        # need a separate loop because `mod.proj` below is a `nn.Linear` too
        for mod in model.modules():
            if isinstance(mod, (LLaMAMLP, CausalSelfAttention)):
                mod.proj.reset_parameters = partial(init_weights, mod.proj, std=(1 / math.sqrt(n_embd) / n_layer))

        if not isinstance(fabric.strategy, FSDPStrategy):
            reset_parameters(model)

    fabric.print(f"{fabric.global_rank} time to init weights: {(time.time()-t):.02f}s", flush=True)


def save_checkpoint(fabric, state, tokenizer_dir, checkpoint_file):
    model = state["model"]
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    fabric.print(f"Saving checkpoint to {str(checkpoint_file)!r}")
    fabric.save(checkpoint_file, state)
    if fabric.global_rank == 0:
        save_hyperparameters(setup, checkpoint_file.parent)
        if tokenizer_dir is not None:
            copy_config_files(tokenizer_dir, checkpoint_file.parent)
        save_config(model.config, checkpoint_file.parent)


def validate_args(train: TrainArgs, eval: EvalArgs, initial_checkpoint_dir, resume) -> None:
    issues = []
    unsupported = [(train, ["epochs"]), (eval, ["max_new_tokens"])]
    for args, names in unsupported:
        for name in names:
            if getattr(args, name) is not None:
                issues.append(f"{__file__} doesn't support the {name!r} argument. This is set in {args}")
    required = [(train, ["max_tokens", "max_norm"])]
    for args, names in required:
        for name in names:
            if getattr(args, name) is None:
                issues.append(f"{__file__} requires the {name!r} argument. This is set in {args}")
    if initial_checkpoint_dir and resume:
        issues.append("Can't provide both `--resume` and `--initial_checkpoint_dir`. Choose one.")
    if issues:
        raise ValueError("\n".join(issues))


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    CLI(setup)
