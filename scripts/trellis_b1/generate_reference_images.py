"""Batch-generate clean object reference images with FLUX via diffusers."""

from __future__ import annotations

import argparse
import inspect
import io
import json
import os
import traceback
from pathlib import Path

import torch


def _is_flux2_model(model: str) -> bool:
    normalized = model.lower()
    return "flux.2" in normalized or "flux2" in normalized


def _iter_prompts(path: Path) -> list[dict[str, object]]:
    prompts: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if not isinstance(data, dict) or not data.get("prompt"):
                raise ValueError(f"Invalid prompt JSONL entry at line {line_number}")
            prompts.append(data)
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompts", type=Path, required=True, help="JSONL with {id,prompt,negative_prompt?}."
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output directory for PNG references."
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("FLUX_MODEL", "black-forest-labs/FLUX.1-dev"),
        help="Diffusers model id.",
    )
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--pipeline",
        choices=("auto", "flux1", "flux2"),
        default="auto",
        help="Pipeline family. auto detects FLUX.2 model ids.",
    )
    parser.add_argument(
        "--flux2-remote-text-encoder",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the hosted FLUX.2 text encoder. Recommended for 24-48 GB GPUs.",
    )
    parser.add_argument(
        "--no-skip-existing", action="store_true", help="Regenerate PNGs even when output exists."
    )
    return parser


def _remote_flux2_text_encoder(prompt: str, device: str) -> torch.Tensor:
    import requests
    from huggingface_hub import get_token

    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        raise RuntimeError("HF_TOKEN is required for the hosted FLUX.2 text encoder.")
    response = requests.post(
        "https://remote-text-encoder-flux-2.huggingface.co/predict",
        json={"prompt": prompt},
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        timeout=120,
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"Remote FLUX.2 text encoder failed with HTTP {response.status_code}: "
            f"{response.text[:500]}"
        )
    return torch.load(io.BytesIO(response.content), map_location=device).to(device)


def _load_pipeline(model: str, use_flux2: bool, use_flux2_remote_text_encoder: bool):
    try:
        if use_flux2:
            from diffusers import Flux2Pipeline

            flux2_kwargs: dict[str, object] = {
                "torch_dtype": torch.bfloat16,
                "token": os.environ.get("HF_TOKEN"),
            }
            if use_flux2_remote_text_encoder:
                flux2_kwargs["text_encoder"] = None
            pipe = Flux2Pipeline.from_pretrained(
                model,
                **flux2_kwargs,
            )
            if use_flux2_remote_text_encoder:
                pipe.to("cuda")
            else:
                pipe.enable_model_cpu_offload()
            return pipe

        from diffusers import FluxPipeline

        pipe = FluxPipeline.from_pretrained(
            model,
            torch_dtype=torch.bfloat16,
            token=os.environ.get("HF_TOKEN"),
        )
        pipe.enable_model_cpu_offload()
        return pipe
    except Exception as exc:
        message = str(exc)
        if "401" in message or "gated" in message.lower() or "authenticated" in message.lower():
            raise RuntimeError(
                f"Cannot download {model}. Set HF_TOKEN and accept the model license on "
                f"Hugging Face."
            ) from exc
        raise


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    prompts = _iter_prompts(args.prompts)
    use_flux2 = args.pipeline == "flux2" or (
        args.pipeline == "auto" and _is_flux2_model(args.model)
    )

    pipe = _load_pipeline(args.model, use_flux2, args.flux2_remote_text_encoder)
    failures: list[dict[str, str]] = []
    failures_path = args.output / "failures.jsonl"
    failures_path.unlink(missing_ok=True)

    for index, item in enumerate(prompts):
        item_id = str(item.get("id") or f"object_{index:04d}")
        image_path = args.output / f"{item_id}.png"
        metadata_path = args.output / f"{item_id}.json"
        if (
            image_path.exists()
            and image_path.stat().st_size > 0
            and metadata_path.exists()
            and not args.no_skip_existing
        ):
            print(f"skipped existing {item_id}")
            continue
        prompt = str(item["prompt"])
        negative_prompt = str(item.get("negative_prompt") or "")
        try:
            generator = torch.Generator("cuda").manual_seed(args.seed + index)
            call_kwargs: dict[str, object] = {
                "height": args.height,
                "width": args.width,
                "guidance_scale": args.guidance,
                "num_inference_steps": args.steps,
                "max_sequence_length": 512,
                "generator": generator,
            }
            if use_flux2 and args.flux2_remote_text_encoder:
                call_kwargs["prompt_embeds"] = _remote_flux2_text_encoder(prompt, "cuda")
            else:
                call_kwargs["prompt"] = prompt
            if (
                not use_flux2
                and negative_prompt
                and "negative_prompt" in inspect.signature(pipe.__call__).parameters
            ):
                call_kwargs["negative_prompt"] = negative_prompt
            image = pipe(**call_kwargs).images[0]
            image.save(image_path)
            metadata = {
                "id": item_id,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "model": args.model,
                "seed": args.seed + index,
                "height": args.height,
                "width": args.width,
                "steps": args.steps,
                "guidance": args.guidance,
            }
            metadata_path.write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(f"generated {item_id}")
        except Exception as exc:  # noqa: BLE001 - keep long production batches moving.
            image_path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
            failure = {
                "id": item_id,
                "prompt": prompt,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            with failures_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(failure, sort_keys=True) + "\n")
            print(f"failed {item_id}: {type(exc).__name__}: {exc}", flush=True)
            torch.cuda.empty_cache()
    if failures:
        print(f"completed with {len(failures)} failed references; see {failures_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
