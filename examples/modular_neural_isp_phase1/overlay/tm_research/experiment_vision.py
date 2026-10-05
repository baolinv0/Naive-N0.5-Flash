"""Actual bounded DEV image inputs and non-authoritative Qwen observations."""
import base64
import copy
import hashlib
import io
import json
from pathlib import Path


MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_TOTAL_IMAGE_BYTES = 16 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
_FORBIDDEN = {"confirm", "confirmation", "test", "tests"}


def _forbidden(path):
    return any(_forbidden_name(part) for part in path.parts)


def _forbidden_name(name):
    normalized = name.casefold()
    return normalized in _FORBIDDEN or normalized.startswith(("confirm_", "confirmation_", "test_"))


def build_vision_messages(packet, image_paths, allowed_root, max_images=3):
    if not isinstance(packet, dict):
        raise ValueError("Feedback packet must be an object")
    if isinstance(max_images, bool) or not isinstance(max_images, int) or not 1 <= max_images <= 3:
        raise ValueError("max_images must be between 1 and 3")
    if not isinstance(image_paths, list) or not 1 <= len(image_paths) <= max_images:
        raise ValueError("Supply between 1 and max_images image paths")
    declared_root = Path(allowed_root).absolute()
    if _forbidden(declared_root):
        raise ValueError("CONFIRM and TEST roots cannot be sent to a model")
    root = declared_root.resolve(strict=True)
    if not root.is_dir() or _forbidden(root) or _forbidden_name(root.name):
        raise ValueError("allowed_root must be an authorized DEV run directory")
    text = {"purpose": "auxiliary DEV visual observations", "feedback_identity": copy.deepcopy(packet), "assets": []}
    content, total = [], 0
    for image_path in image_paths:
        path = Path(image_path)
        if not path.is_absolute():
            path = root / path
        if _forbidden(path):
            raise ValueError("CONFIRM and TEST assets cannot be sent to a model")
        path = path.resolve(strict=True)
        if not path.is_relative_to(root) or _forbidden(path) or not path.is_file():
            raise ValueError("Image must remain within authorized DEV root")
        if any(_forbidden_name(part) for part in path.relative_to(root).parts):
            raise ValueError("CONFIRM and TEST assets cannot be sent to a model")
        if path.suffix.casefold() not in (".png", ".jpg", ".jpeg"):
            raise ValueError("Only PNG and JPEG assets are allowed")
        with path.open("rb") as stream:
            data = stream.read(MAX_IMAGE_BYTES + 1)
        total += len(data)
        if not data or len(data) > MAX_IMAGE_BYTES or total > MAX_TOTAL_IMAGE_BYTES:
            raise ValueError("Image byte budget exceeded")
        from PIL import Image
        try:
            with Image.open(io.BytesIO(data)) as image:
                image_format = image.format
                dimensions = list(image.size)
                if image_format not in ("PNG", "JPEG") or image.width * image.height > MAX_IMAGE_PIXELS:
                    raise ValueError("Unsupported image format or dimensions")
                image.verify()
            # JPEG verify() checks headers but can miss truncated pixel payloads.
            with Image.open(io.BytesIO(data)) as decoded:
                decoded.load()
        except Exception as exc:
            raise ValueError("Asset is not a valid bounded PNG/JPEG") from exc
        mime = "image/png" if image_format == "PNG" else "image/jpeg"
        text["assets"].append({"path": str(path), "sha256": hashlib.sha256(data).hexdigest(),
                               "bytes": len(data), "dimensions": dimensions, "mime_type": mime})
        content.append({"type": "image_url", "image_url": {
            "url": "data:" + mime + ";base64," + base64.b64encode(data).decode("ascii")}})
    content.insert(0, {"type": "text", "text": json.dumps(text, ensure_ascii=False, allow_nan=False)})
    return [{"role": "system", "content":
             "Describe only visible DEV image observations and limitations. Return a JSON object with "
             "observations and limitations, each an array of strings. You provide auxiliary descriptions; "
             "do not assign PSNR, select checkpoints, accept proposals, or change budgets."},
            {"role": "user", "content": content}]


def validate_auxiliary_report(payload):
    if not isinstance(payload, dict) or set(payload) != {"observations", "limitations"}:
        raise ValueError("Auxiliary report requires only observations and limitations")
    for field in ("observations", "limitations"):
        values = payload[field]
        if not isinstance(values, list) or len(values) > 100 or any(
                not isinstance(value, str) or not value.strip() or len(value) > 4000 for value in values):
            raise ValueError(field + " must be a bounded array of nonempty strings")
    return {**copy.deepcopy(payload), "authority": "auxiliary_only"}
