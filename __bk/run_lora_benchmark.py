import os
import json
import argparse
import httpx
from pathlib import Path
import time
import random

TEMPLATE_GRAPH_PATH = "benchmark.json"

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
        raise RuntimeError(f"Request failed with status {response.status_code}: {response.text}")
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
        lora_names = lora_loader_info.get("input", {}).get("required", {}).get("lora_name", [[]])[0]
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
    

    for idx, raw_prompt in enumerate(prompts, start=1):
        prompt = f"{args.trigger_word}, {raw_prompt}" if args.trigger_word else raw_prompt
        tag = f"prompt_{idx:03}"
        output_prefix = f"benchmark_{tag}"
        
        # Set the seed: use fixed_seed if provided, otherwise derive from prompt
        prompt_seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)

        # Define combinations of LoRA configurations
        combos = [("no_lora", None, None), ("char_lora", args.char_lora, None)]
        if args.style_lora:
            combos.append(("char_plus_style", args.char_lora, args.style_lora))

        print(f"🧪 Running {tag} with seed:{prompt_seed}")

        for label, char_path, style_path in combos:
            # Load the template graph
            with open(TEMPLATE_GRAPH_PATH, "r") as f:
                full_graph = json.load(f)

            graph = full_graph.get("prompt", full_graph)
            id_to_node = graph  # Graph is a dict with node IDs as keys

            # Identify nodes by class_type and title
            prompt_node = next((k for k, v in id_to_node.items() if v.get("class_type") in ("CLIPTextEncode", "CLIPTextEncodeSDXL")), None)
            seed_node = next((k for k, v in id_to_node.items() if v.get("class_type") in ("KSamplerAdvanced", "KSampler")), None)
            outprefix_node = next((k for k, v in id_to_node.items() if v.get("class_type", "").lower().startswith("saveimage")), None)
            char_lora_node = next((k for k, v in id_to_node.items() if v.get("class_type") in ("LoraLoader", "LoraLoaderModelOnly") and v.get("_meta", {}).get("title", "").lower().startswith("my_lora")), None)
            style_lora_node = next((k for k, v in id_to_node.items() if v.get("class_type") in ("LoraLoader", "LoraLoaderModelOnly") and v.get("_meta", {}).get("title", "").lower().startswith("style_lora")), None)

            # Update node inputs correctly
            if prompt_node:
                id_to_node[prompt_node]["inputs"]["text"] = prompt
            if seed_node:
                id_to_node[seed_node]["inputs"]["seed"] = prompt_seed
            if outprefix_node:
                id_to_node[outprefix_node]["inputs"]["filename_prefix"] = f"{output_prefix}_{label}"
            if char_lora_node:
                node = id_to_node[char_lora_node]
                node["inputs"]["lora_name"] = char_path if char_path else default_lora
                node["inputs"]["strength_model"] = 1.0 if char_path else 0.0
            if style_lora_node:
                node = id_to_node[style_lora_node]
                node["inputs"]["lora_name"] = style_path if style_path else default_lora
                node["inputs"]["strength_model"] = args.style_weight if style_path else 0.0

            # Debug output
            print(f"🧪 Running {label} for {tag}...")
            #debug_path = f"/tmp/graph_debug_{tag}_{label}.json"
            #with open(debug_path, "w") as dbg:
            #    json.dump(graph, dbg, indent=2)
            #print(f"🛠️ Saved debug graph to {debug_path}")

            # Send to ComfyUI and wait for completion
            res = send_to_comfy_http(args.host, {"prompt": graph})
            prompt_id = res.get("prompt_id")
            if prompt_id:
                wait_for_execution(args.host, prompt_id)
            print(f"✅ Completed: {output_prefix}_{label}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA Benchmarking Tool for ComfyUI")
    parser.add_argument("--prompts", type=str, required=True, help="Path to prompts.txt")
    parser.add_argument("--char_lora", type=str, required=True, help="Relative path to character LoRA (e.g., FLUX/eiza/eiza_dev_v06.safetensors)")
    parser.add_argument("--style_lora", type=str, help="Relative path to style LoRA")
    parser.add_argument("--style_weight", type=float, default=0.7, help="Style LoRA weight")
    parser.add_argument("--host", type=str, default="http://192.168.10.130:8188", help="HTTP API URL for ComfyUI")
    parser.add_argument("--trigger_word", type=str, default=None, help="Optional trigger word to prepend to prompts")
    parser.add_argument("--seed", type=int, default=None, help="Optional fixed seed for ALL generations")

    args = parser.parse_args()
    run_benchmark(args)
