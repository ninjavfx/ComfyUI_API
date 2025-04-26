import os
import sys
import json
import argparse
import httpx
import time
import random

TEMPLATE_GRAPH_PATH = "benchmark.json"


def check_comfy(url, json):
    try:
        return httpx.post(url, json=json, timeout=5)
    except httpx.RequestError:
        # This includes ConnectError, Timeout, etc.
        print("❌ Can't access ComfyUI. Make sure the server is running")
        sys.exit(1)


def send_to_comfy_http(host, graph):
    """Send the graph to ComfyUI via HTTP API."""
    if not isinstance(graph, dict):
        raise ValueError("Graph is not a dictionary")

    if "nodes" in graph:
        graph = {"prompt": graph}
    elif "prompt" not in graph:
        raise ValueError("Payload is missing 'prompt' key")

    url = f"{host}/prompt"  # Correct endpoint for ComfyUI API
    response = httpx.post(url, json=graph)
    if response.status_code != 200:
        debug_path = "/tmp/failure_graph.json"
        with open(debug_path, "w") as dbg:
            json.dump(graph, dbg, indent=2)
        print(f"❌ Request failed. Graph saved to {debug_path}")
        raise RuntimeError(
            f"Request failed with status {response.status_code}: {response.text}"
        )
    return response.json()


def wait_for_execution(host, prompt_id, interval=1.5, timeout=120):
    """Wait for the execution of the prompt to complete."""
    status_url = f"{host}/history/{prompt_id}"
    for _ in range(int(timeout / interval)):
        try:
            res = httpx.get(status_url)
            if res.status_code == 200 and "outputs" in res.json().get(prompt_id, {}):
                return res.json()
        except Exception:
            pass
        time.sleep(interval)
    raise TimeoutError(f"Timeout waiting for prompt {prompt_id} to finish")


def get_available_loras(host):
    """Fetch the list of available LoRAs from ComfyUI."""
    response = httpx.get(f"{host}/object_info")
    if response.status_code == 200:
        object_info = response.json()
        lora_loader_info = object_info.get("LoraLoaderModelOnly", {})
        lora_names = (
            lora_loader_info.get("input", {})
            .get("required", {})
            .get("lora_name", [[]])[0]
        )
        return lora_names
    else:
        raise RuntimeError(f"Failed to fetch object info: {response.status_code}")


def run_benchmark(args):
    """Run the benchmark with different LoRA configurations."""

    # Fetch available LoRAs and set a default
    available_loras = get_available_loras(args.host)
    if not available_loras:
        raise ValueError("No LoRAs available in ComfyUI")
    default_lora = available_loras[0]

    # Validate provided LoRA names
    if args.char_lora not in available_loras:
        raise ValueError(f"Character LoRA {args.char_lora} not found")
    if args.style_lora and args.style_lora not in available_loras:
        raise ValueError(f"Style LoRA {args.style_lora} not found")

    # Load prompts from file
    with open(args.prompts, "r") as f:
        prompts = [p.strip() for p in f.read().split("\n\n") if p.strip()]

    # Check dependency: if one is set, the other must also be set
    if (args.width is None) != (args.height is None):
        parser.error("Both --width and --height must be specified together.")

    # Get resolution if set
    width = args.width if args.width is not None else 1024
    height = args.height if args.height is not None else 1024

    # TODO: Set output path as argument
    # Set output path
    output_path = os.path.join(os.getcwd(), "output")
    os.makedirs(output_path, exist_ok=True)

    for idx, raw_prompt in enumerate(prompts, start=1):
        prompt = (
            f"{args.trigger_word}, {raw_prompt}" if args.trigger_word else raw_prompt
        )
        # Simplified prompt tag: p_001, p_002, etc.
        tag = f"p_{idx:03}"
        tag_prompt = f"prompt_{idx:03}"

        # Get character LoRA name without extension for filename
        char_lora_name = os.path.splitext(os.path.basename(args.char_lora))[0]

        # Define combinations of LoRA configurations
        if args.only_lora:
            if args.save_no_style:
                # Only character LoRA, no style LoRA
                combos = [("char_lora", args.char_lora, None)]
            else:
                # Skip the character-only generation when save_no_style is False
                combos = []
        else:
            if args.save_no_style:
                # Include both no LoRA and character-only LoRA
                combos = [("no_lora", None, None), ("char_lora", args.char_lora, None)]
            else:
                # Only include no LoRA case (skip character-only)
                combos = [("no_lora", None, None)]

        # Always include the character+style combo if style_lora is provided
        if args.style_lora:
            combos.append(("char_plus_style", args.char_lora, args.style_lora))

        # Repeat the generation as specified by the --repeats option
        for repeat_idx in range(1, args.repeats + 1):
            # Set a unique seed for each repeat unless fixed seed is provided
            if args.seed is not None:
                prompt_seed = args.seed
            else:
                # Generate a 15-digit seed (using a larger range than 2**32)
                prompt_seed = random.randint(10**14, 10**15 - 1)

            # Create a unique output prefix for each repeat with formatted repeat number
            repeat_str = f"r_{repeat_idx:02}"

            print(
                f"🧪 Running {tag_prompt} repeat {repeat_idx}/{args.repeats} with seed:{prompt_seed}"
            )

            for label, char_path, style_path in combos:
                # Load the template graph
                with open(TEMPLATE_GRAPH_PATH, "r") as f:
                    full_graph = json.load(f)

                graph = full_graph.get("prompt", full_graph)
                id_to_node = graph  # Graph is a dict with node IDs as keys

                # Identify nodes by class_type and title
                prompt_node = next(
                    (
                        k
                        for k, v in id_to_node.items()
                        if v.get("class_type") in ("CLIPTextEncode")
                    ),
                    None,
                )
                scheduler_node = next(
                    (
                        k
                        for k, v in id_to_node.items()
                        if v.get("class_type") in ("BasicScheduler")
                    ),
                    None,
                )
                noise_node = next(
                    (
                        k
                        for k, v in id_to_node.items()
                        if v.get("class_type") in ("RandomNoise")
                    ),
                    None,
                )
                save_image_node = next(
                    (
                        k
                        for k, v in id_to_node.items()
                        if v.get("class_type", "").lower().startswith("saveimage")
                    ),
                    None,
                )
                char_lora_node = next(
                    (
                        k
                        for k, v in id_to_node.items()
                        if v.get("class_type") in ("LoraLoaderModelOnly")
                        and v.get("_meta", {})
                        .get("title", "")
                        .lower()
                        .startswith("my_lora")
                    ),
                    None,
                )
                style_lora_node = next(
                    (
                        k
                        for k, v in id_to_node.items()
                        if v.get("class_type") in ("LoraLoaderModelOnly")
                        and v.get("_meta", {})
                        .get("title", "")
                        .lower()
                        .startswith("style_lora")
                    ),
                    None,
                )
                empty_latent_node = next(
                    (
                        k
                        for k, v in id_to_node.items()
                        if v.get("class_type") in ("EmptySD3LatentImage")
                    ),
                    None,
                )
                model_sampling_node = next(
                    (
                        k
                        for k, v in id_to_node.items()
                        if v.get("class_type") in ("ModelSamplingFlux")
                    ),
                    None,
                )
                flux_guidance_node = next(
                    (
                        k
                        for k, v in id_to_node.items()
                        if v.get("class_type") in ("FluxGuidance")
                    ),
                    None,
                )

                # Update node inputs correctly
                if prompt_node:
                    id_to_node[prompt_node]["inputs"]["text"] = prompt
                if scheduler_node:
                    id_to_node[scheduler_node]["inputs"]["steps"] = (
                        args.steps if args.steps is not None else 25
                    )
                if noise_node:
                    id_to_node[noise_node]["inputs"]["noise_seed"] = prompt_seed
                if save_image_node:
                    node = id_to_node[save_image_node]
                    # Map the old label names to new naming convention
                    if label == "no_lora":
                        label_code = "nl"
                    elif label == "char_lora":
                        label_code = "c"
                    elif label == "char_plus_style":
                        label_code = "cs"
                    else:
                        label_code = label

                    # Construct the new filename format
                    # Format: p_001_character-name_r_01_seed_labelcode
                    filename = f"{tag}_{char_lora_name}_{repeat_str}_{prompt_seed}_{label_code}"
                    node["inputs"]["filename_prefix"] = filename
                    node["inputs"]["output_folder"] = f"{output_path}"
                if char_lora_node:
                    node = id_to_node[char_lora_node]
                    node["inputs"]["lora_name"] = (
                        char_path if char_path else default_lora
                    )
                    node["inputs"]["strength_model"] = (
                        args.lora_weight if char_path else 0.0
                    )
                if style_lora_node:
                    node = id_to_node[style_lora_node]
                    node["inputs"]["lora_name"] = (
                        style_path if style_path else default_lora
                    )
                    node["inputs"]["strength_model"] = (
                        args.style_weight if style_path else 0.0
                    )
                if empty_latent_node:
                    node = id_to_node[empty_latent_node]
                    node["inputs"]["width"] = width
                    node["inputs"]["height"] = height

                if model_sampling_node:
                    node = id_to_node[model_sampling_node]
                    node["inputs"]["width"] = width
                    node["inputs"]["height"] = height

                if flux_guidance_node:
                    id_to_node[flux_guidance_node]["inputs"]["guidance"] = (
                        args.flux_guidance if args.flux_guidance is not None else 3.5
                    )

                # Debug output
                print(
                    f"🧪 Running {label} for {tag_prompt} (repeat {repeat_idx}/{args.repeats})..."
                )
                # debug_path = f"/tmp/graph_debug_{tag}_{label}.json"
                # with open(debug_path, "w") as dbg:
                #    json.dump(graph, dbg, indent=2)
                # print(f"🛠️ Saved debug graph to {debug_path}")

                # Send to ComfyUI and wait for completion
                res = send_to_comfy_http(args.host, {"prompt": graph})
                prompt_id = res.get("prompt_id")
                if prompt_id:
                    wait_for_execution(args.host, prompt_id)
                print(f"✅ Completed: {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA Benchmarking Tool for ComfyUI")
    parser.add_argument(
        "--prompts", type=str, required=True, help="Path to prompts.txt"
    )
    parser.add_argument(
        "--only_lora",
        action="store_true",
        help="Enable only_lora mode, default is off which means the first generation is with no Loras",
    )
    parser.add_argument(
        "--char_lora",
        type=str,
        required=True,
        help="Relative path to character LoRA (e.g., FLUX/my_lora/my_lora.safetensors)",
    )
    parser.add_argument(
        "--lora_weight", type=float, default=1.0, help="Main LoRA weight. Default 1.0"
    )
    parser.add_argument("--style_lora", type=str, help="Relative path to style LoRA")
    parser.add_argument(
        "--style_weight", type=float, default=0.7, help="Style LoRA weight"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="http://127.0.0.1:8188",
        help="HTTP API URL for ComfyUI",
    )
    parser.add_argument(
        "--trigger_word",
        type=str,
        default=None,
        help="Optional trigger word to prepend to prompts",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Optional fixed seed for ALL generations"
    )
    parser.add_argument("--width", type=int, help="Override image width. Default 1024")
    parser.add_argument(
        "--height", type=int, help="Override image height. Default 1024"
    )
    parser.add_argument("--steps", type=int, help="Set the number of steps. Default 25")
    parser.add_argument(
        "--flux_guidance",
        type=float,
        help="Set the value for FLuxGuidance. Default 3.5",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of times to repeat each prompt generation",
    )
    parser.add_argument(
        "--save_no_style",
        action="store_true",
        help="If set, generate and save images without style LoRA (character LoRA only)",
    )

    args = parser.parse_args()

    # Is Comfy running?
    url = f"{args.host}/prompt"
    check_comfy(url, json={"prompt": []})

    run_benchmark(args)
