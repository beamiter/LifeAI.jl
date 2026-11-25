#!/usr/bin/env python3
"""Export an official Transformers 4.57.x Qwen3-VL video reference oracle."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import save_file
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def make_synthetic_video(num_frames: int = 4, height: int = 96, width: int = 128) -> np.ndarray:
    """Deterministic RGB uint8 video (T, H, W, 3) with distinct per-frame motion."""
    frames = np.zeros((num_frames, height, width, 3), dtype=np.uint8)
    ys = np.arange(height, dtype=np.int32)[:, None]
    xs = np.arange(width, dtype=np.int32)[None, :]
    for t in range(num_frames):
        # Moving bright square across frames on a structured gradient background.
        frames[t, :, :, 0] = ((xs * 2 + t * 37) % 256).astype(np.uint8)
        frames[t, :, :, 1] = ((ys * 2 + t * 53) % 256).astype(np.uint8)
        frames[t, :, :, 2] = ((xs + ys + t * 71) % 256).astype(np.uint8)
        y0 = 12 + t * 14
        x0 = 16 + t * 20
        frames[t, y0:y0 + 24, x0:x0 + 24, 0] = 240
        frames[t, y0:y0 + 24, x0:x0 + 24, 1] = 30 + t * 50
        frames[t, y0:y0 + 24, x0:x0 + 24, 2] = 20
    return frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--greedy-tokens", type=int, default=8)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    video = make_synthetic_video(num_frames=args.num_frames, height=96, width=128)
    np.save(out / "video_frames.npy", video)

    processor = AutoProcessor.from_pretrained(args.model_dir)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": "Briefly describe the moving object in this video."},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    video_metadata = [
        {
            "fps": args.fps,
            "frames_indices": list(range(args.num_frames)),
            "total_num_frames": args.num_frames,
        }
    ]
    inputs = processor(
        text=[prompt],
        videos=[video],
        video_metadata=video_metadata,
        return_tensors="pt",
    )

    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_dir,
        torch_dtype=dtype,
        attn_implementation="eager",
    ).to(args.device)
    model.eval()

    model_inputs = {k: v.to(args.device) for k, v in inputs.items()}
    if "pixel_values_videos" in model_inputs:
        model_inputs["pixel_values_videos"] = model_inputs["pixel_values_videos"].to(dtype)
    with torch.no_grad():
        pos_ids, rope_deltas = model.model.get_rope_index(
            model_inputs["input_ids"],
            video_grid_thw=model_inputs["video_grid_thw"],
            attention_mask=model_inputs["attention_mask"],
        )
        gen_out = model.generate(
            **model_inputs,
            do_sample=False,
            max_new_tokens=args.greedy_tokens,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
        )
        prompt_len = model_inputs["input_ids"].shape[1]
        generated = [int(t) for t in gen_out.sequences[0, prompt_len:].tolist()]
        step_logits = [s[0].detach().cpu().to(torch.float32) for s in gen_out.scores]

    tensors = {
        "video_frames": torch.from_numpy(video),
        "pixel_values_videos": inputs["pixel_values_videos"].detach().cpu().to(torch.float32),
        "video_grid_thw": inputs["video_grid_thw"].detach().cpu().to(torch.int64),
        "input_ids": inputs["input_ids"].detach().cpu().to(torch.int64),
        "position_ids": pos_ids.detach().cpu().to(torch.int64),
        "rope_deltas": rope_deltas.detach().cpu().to(torch.int64),
    }
    for idx, logits in enumerate(step_logits):
        tensors[f"step{idx}_logits"] = logits
    tensor_path = out / "reference.safetensors"
    save_file(tensors, str(tensor_path))
    tensor_sha = hashlib.sha256(tensor_path.read_bytes()).hexdigest()

    meta = {
        "model_dir": str(Path(args.model_dir).resolve()),
        "dtype": args.dtype,
        "device": args.device,
        "fps": args.fps,
        "num_frames": args.num_frames,
        "height": 96,
        "width": 128,
        "prompt": prompt,
        "prompt_length": int(inputs["input_ids"].shape[1]),
        "video_grid_thw": inputs["video_grid_thw"].tolist(),
        "rope_deltas": rope_deltas.tolist(),
        "greedy_ids_zero_based": generated,
        "completion": processor.tokenizer.decode(generated, skip_special_tokens=True),
        "tensor_sha256": tensor_sha,
    }
    meta_path = out / "reference.json"
    meta_text = json.dumps(meta, indent=2, ensure_ascii=False) + "\n"
    meta_path.write_text(meta_text)
    print(json.dumps({
        "output_dir": str(out),
        "prompt_length": meta["prompt_length"],
        "video_grid_thw": meta["video_grid_thw"],
        "rope_deltas": meta["rope_deltas"],
        "greedy_ids_zero_based": generated,
        "completion": meta["completion"],
        "tensor_sha256": tensor_sha,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
