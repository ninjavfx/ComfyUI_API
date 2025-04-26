import argparse
import json
import sys
from PIL import Image, PngImagePlugin
import re


def extract_info_from_png(image_path):
    """Extract the text prompt, seed, and image size from a PNG file created by Flux diffusion model."""
    try:
        # Open the image
        img = Image.open(image_path)

        # Check if it's a PNG
        if img.format != "PNG":
            print(f"Error: {image_path} is not a PNG file", file=sys.stderr)
            return None, None, None

        # Get image size
        image_size = f"{img.width}x{img.height}"

        # Extract metadata
        metadata = img.info

        # Look for text prompt and seed in the metadata
        prompt = None
        seed = None

        # Try to find JSON data in the metadata
        for key, value in metadata.items():
            if isinstance(value, str):
                try:
                    # Try to parse as JSON
                    json_data = json.loads(value)

                    # Look for the specific structure with "text" key
                    for node_key, node_data in json_data.items():
                        if isinstance(node_data, dict) and "inputs" in node_data:
                            inputs = node_data["inputs"]

                            # Look for text prompt
                            if isinstance(inputs, dict) and "text" in inputs:
                                prompt = inputs["text"]

                            # Look for seed value
                            if isinstance(inputs, dict) and "noise_seed" in inputs:
                                seed = inputs["noise_seed"]

                except json.JSONDecodeError:
                    # Not valid JSON, try with regex
                    if prompt is None:
                        text_match = re.search(
                            r'"inputs":\s*{\s*"text":\s*"([^"]+)"', value
                        )
                        if text_match:
                            prompt = text_match.group(1)

                    if seed is None:
                        seed_match = re.search(
                            r'"inputs":\s*{\s*"noise_seed":\s*(\d+)', value
                        )
                        if seed_match:
                            seed = int(seed_match.group(1))

        # If we couldn't find the prompt in the expected structure, look for any "text" field
        if prompt is None:
            for key, value in metadata.items():
                if isinstance(value, str):
                    text_match = re.search(r'"text":\s*"([^"]+)"', value)
                    if text_match:
                        prompt = text_match.group(1)

        # If we couldn't find the seed in the expected structure, look for any "noise_seed" field
        if seed is None:
            for key, value in metadata.items():
                if isinstance(value, str):
                    seed_match = re.search(r'"noise_seed":\s*(\d+)', value)
                    if seed_match:
                        seed = int(seed_match.group(1))

        return prompt, seed, image_size

    except Exception as e:
        print(f"Error processing {image_path}: {e}", file=sys.stderr)
        return None, None, None


def main():
    parser = argparse.ArgumentParser(
        description="Extract text prompts and seed from Flux diffusion PNG metadata"
    )
    parser.add_argument(
        "images",
        metavar="IMAGE",
        type=str,
        nargs="+",
        help="PNG image file(s) to process",
    )
    parser.add_argument(
        "--output", "-o", type=str, help="Output file (default: stdout)"
    )

    args = parser.parse_args()

    output_file = open(args.output, "w") if args.output else sys.stdout

    for image_path in args.images:
        prompt, seed, image_size = extract_info_from_png(image_path)

        if len(args.images) > 1:
            print(f"File: {image_path}", file=output_file)

        print("Prompt:", file=output_file)
        print(prompt if prompt else "Not found", file=output_file)
        print("", file=output_file)
        print(f"Seed: {seed if seed is not None else 'Not found'}", file=output_file)
        print(f"Image size: {image_size}", file=output_file)

        if len(args.images) > 1:
            print("\n" + "-" * 50 + "\n", file=output_file)

    if args.output:
        output_file.close()


if __name__ == "__main__":
    main()

