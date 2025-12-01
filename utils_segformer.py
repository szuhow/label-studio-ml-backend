import os
import pickle
from typing import Any, Dict, Optional, Tuple
import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor, SegformerConfig
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import json
import numpy as np

# Sprawdzenie dostępności GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'PyTorch używa GPU: {torch.cuda.get_device_name(0)}')
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("PyTorch używa MPS (Metal Performance Shaders)")
else:
    device = torch.device("cpu")
    print("GPU niedostępne, używamy CPU")

print(f" Device: {device}")
# path = '/home/ives/rafal/notebooks/data'

# Mapa dostępnych modeli HF → doboru bazowej architektury
AVAILABLE_MODELS: Dict[str, list[str]] = {
    "b0": ["256-256", "512-512"],
    "b1": ["512-512"],
    "b2": ["512-512"],
    "b3": ["512-512"],
    "b4": ["512-512"],
    "b5": ["640-640"],  # B5 ma tylko 640
}

def find_checkpoint_path(checkpoint_path: Optional[str] = None, search_dir: str = ".") -> str:
    if checkpoint_path and os.path.isfile(checkpoint_path):
        return checkpoint_path

    candidates = [f for f in os.listdir(search_dir) if f.endswith((".pth", ".pt"))]
    if not candidates:
        raise FileNotFoundError("No checkpoint (.pth/.pt) found")

    # preferuj pliki z 'best' w nazwie
    best = [f for f in candidates if "best" in f.lower()]
    chosen = best[0] if best else candidates[0]
    return os.path.join(search_dir, chosen)


def get_segformer_model_mapping(model_size: str, requested_image_size: int) -> Tuple[str, int]:
    if model_size not in AVAILABLE_MODELS:
        raise ValueError(f"Unsupported model size: {model_size}")
    available_sizes = AVAILABLE_MODELS[model_size]
    req = f"{requested_image_size}-{requested_image_size}"
    if req in available_sizes:
        model_name = f"nvidia/segformer-{model_size}-finetuned-ade-{requested_image_size}-{requested_image_size}"
        return model_name, requested_image_size
    ints = [int(s.split("-")[0]) for s in available_sizes]
    closest = min(ints, key=lambda x: abs(x - requested_image_size))
    model_name = f"nvidia/segformer-{model_size}-finetuned-ade-{closest}-{closest}"
    return model_name, requested_image_size


def _load_raw_checkpoint(path: str):
    try:
        # First try with weights_only=True (safer)
        return torch.load(path, map_location="cpu", weights_only=True)
    except (TypeError, pickle.UnpicklingError):
        # If that fails, try without weights_only (for custom classes)
        print(f"Warning: Loading checkpoint with weights_only=False (less secure)")
        return torch.load(path, map_location="cpu", weights_only=False)


def _extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    # Handle SegFormerCheckpoint object
    if hasattr(ckpt, 'model_state_dict') and isinstance(ckpt.model_state_dict, dict):
        return ckpt.model_state_dict
    
    # Handle dictionary format
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
            return ckpt["model_state_dict"]
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"]
        if ckpt and all(isinstance(k, str) for k in ckpt.keys()) and all(torch.is_tensor(v) for v in ckpt.values()):
            return ckpt
    return {}


def load_model_for_inference_from_checkpoint(
    checkpoint_path: Optional[str],
    model_size: str,
    image_size: int,
    num_classes: int = 2,
    device: Optional[torch.device] = None,
) -> Tuple[SegformerForSemanticSegmentation, SegformerImageProcessor]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = find_checkpoint_path(checkpoint_path)
    ckpt = _load_raw_checkpoint(ckpt_path)
    state = _extract_state_dict(ckpt)

    if not state:
        raise ValueError(f"No state_dict found in checkpoint: {ckpt_path}")

    # Automatycznie wykryj liczbę klas z checkpointu jeśli nie podano
    if num_classes == 2:
        # Sprawdź kształt classifier w state_dict
        classifier_weight_key = None
        classifier_bias_key = None
        
        for key in state.keys():
            if 'decode_head.classifier.weight' in key or 'classifier.weight' in key:
                classifier_weight_key = key
            elif 'decode_head.classifier.bias' in key or 'classifier.bias' in key:
                classifier_bias_key = key
        
        detected_num_classes = num_classes
        if classifier_weight_key:
            detected_num_classes = state[classifier_weight_key].shape[0]
        elif classifier_bias_key:
            detected_num_classes = state[classifier_bias_key].shape[0]
        
        if detected_num_classes != num_classes and detected_num_classes > 2:
            print(f"Auto-detected num_classes={detected_num_classes} from checkpoint (was {num_classes})")
            num_classes = detected_num_classes

    model_size = model_size.lower()
    base_model_name, processor_size = get_segformer_model_mapping(model_size, int(image_size))

    processor = SegformerImageProcessor.from_pretrained(base_model_name)
    try:
        processor.size = {"height": processor_size, "width": processor_size}
    except Exception:
        processor.size = processor_size
    processor.do_reduce_labels = False

    # Create label mappings for multiclass
    if num_classes > 2:
        # For ARCADE multiclass: create mappings for all classes
        label2id = {i: f"class_{i}" for i in range(num_classes)}
        label2id[0] = "background"
        # Add some meaningful names for common vessel segments
        vessel_names = {
            1: "vessel_1", 2: "vessel_2", 3: "vessel_3", 4: "vessel_4", 5: "vessel_5",
            6: "vessel_6", 7: "vessel_7", 8: "vessel_8", 9: "vessel_9", 10: "vessel_9a",
            11: "vessel_10", 12: "vessel_10a", 13: "vessel_11", 14: "vessel_12", 15: "vessel_12a",
            16: "vessel_13", 17: "vessel_14", 18: "vessel_14a", 19: "vessel_15", 20: "vessel_16",
            21: "vessel_16a", 22: "vessel_16b", 23: "vessel_16c", 24: "vessel_12b", 25: "vessel_14b",
            26: "stenosis"
        }
        for class_id, name in vessel_names.items():
            if class_id < num_classes:
                label2id[class_id] = name
    else:
        # Binary segmentation
        label2id = {0: "background", 1: "vessel"}
    
    id2label = {v: k for k, v in label2id.items()}
    print(f"id2label: {id2label}")
    
    # Pobierz konfigurację bazowego modelu i zmodyfikuj dla naszej liczby klas
    config = SegformerConfig.from_pretrained(base_model_name)
    config.num_labels = num_classes
    config.id2label = {k: v for v, k in id2label.items()}
    config.label2id = label2id
    
    # Utwórz model z konfiguracji (bez wstępnie wytrenowanych wag - unikamy meta tensorów)
    print(f"Creating model from config with {num_classes} classes...")
    model = SegformerForSemanticSegmentation(config)
    
    # Załaduj state_dict z checkpointu
    missing_keys, unexpected_keys = model.load_state_dict(state, strict=False)
    
    if missing_keys:
        print(f"Warning: Missing keys when loading state_dict: {len(missing_keys)} keys")
        # Pokaż kilka brakujących kluczy
        if len(missing_keys) <= 10:
            print(f"  Missing: {missing_keys}")
        else:
            print(f"  First 10 missing: {missing_keys[:10]}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys when loading state_dict: {len(unexpected_keys)} keys")
    
    # Przenieś na docelowe urządzenie i ustaw eval mode
    model = model.to(device)
    model.eval()
    print(f"Model loaded successfully on {device}")
    
    return model, processor

def apply_clahe_pil(
    image: "PIL.Image.Image",
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
    space: str = "lab",  # 'lab' | 'ycrcb' | 'gray'
) -> "PIL.Image.Image":
    """
    Zastosuj CLAHE do obrazu PIL i zwróć obraz PIL RGB.
    """
    arr = np.array(image)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)

    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if space.lower() == "lab":
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l2 = clahe.apply(l)
        lab2 = cv2.merge((l2, a, b))
        bgr_out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    elif space.lower() == "ycrcb":
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        y2 = clahe.apply(y)
        ycrcb2 = cv2.merge((y2, cr, cb))
        bgr_out = cv2.cvtColor(ycrcb2, cv2.COLOR_YCrCb2BGR)
    elif space.lower() == "gray":
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        g2 = clahe.apply(gray)
        bgr_out = cv2.cvtColor(g2, cv2.COLOR_GRAY2BGR)
    else:
        bgr_out = bgr  # brak zmian

    rgb = cv2.cvtColor(bgr_out, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def apply_denoise_pil(
    image: "PIL.Image.Image",
    method: str = "nlmeans",        # 'nlmeans' | 'bilateral' | 'gaussian' | 'median' | 'none'
    # NLMeans params:
    nl_h: int = 2,
    nl_hColor: int = 5,
    nl_template: int = 5,
    nl_search: int = 21,
    # Bilateral params:
    bilateral_d: int = 9,
    bilateral_sigma_color: float = 75.0,
    bilateral_sigma_space: float = 75.0,
    # Gaussian params:
    gaussian_ksize: int = 3,
    gaussian_sigma: float = 0.0,
    # Median params:
    median_ksize: int = 3,
) -> "PIL.Image.Image":
    """
    Redukcja szumu dla obrazu PIL RGB, zwraca PIL RGB.
    """
    arr = np.array(image)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)

    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    m = method.lower()

    if m == "nlmeans":
        # szybki NLMeans dla kolorów
        bgr_out = cv2.fastNlMeansDenoisingColored(
            bgr,
            None,
            h=nl_h,
            hColor=nl_hColor,
            templateWindowSize=max(3, nl_template | 1),  # wymuś nieparzyste
            searchWindowSize=max(7, nl_search | 1),
        )
    elif m == "bilateral":
        bgr_out = cv2.bilateralFilter(
            bgr, d=max(1, bilateral_d), sigmaColor=bilateral_sigma_color, sigmaSpace=bilateral_sigma_space
        )
    elif m == "gaussian":
        k = max(1, gaussian_ksize)
        if k % 2 == 0:
            k += 1
        bgr_out = cv2.GaussianBlur(bgr, (k, k), sigmaX=gaussian_sigma)
    elif m == "median":
        k = max(1, median_ksize)
        if k % 2 == 0:
            k += 1
        bgr_out = cv2.medianBlur(bgr, k)
    else:
        bgr_out = bgr  # brak zmian

    rgb = cv2.cvtColor(bgr_out, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


@torch.no_grad()
def predict_pil_image(
    model: SegformerForSemanticSegmentation,
    image: "PIL.Image.Image",
    processor: SegformerImageProcessor,
    device: Optional[torch.device] = None,
    return_probs: bool = True,
    # Denoise:
    use_denoise: bool = False,
    denoise_method: str = "nlmeans",   # 'nlmeans' | 'bilateral' | 'gaussian' | 'median'
    # CLAHE:
    use_clahe: bool = False,
    clahe_space: str = "lab",
    clahe_clip: float = 2.0,
    clahe_grid: tuple[int, int] = (8, 8),
) -> torch.Tensor:
    """
    Inferencja bezpośrednio na obiekcie PIL.Image.
    Zwraca maskę klas (H, W) jako tensor CPU (int64). Gdy return_probs=True, zwraca także mapę softmax (C,H,W).
    """
    if device is None:
        device = next(model.parameters()).device

    # Preprocessing: najpierw denoise, potem CLAHE
    img_in = image
    if use_denoise:
        img_in = apply_denoise_pil(
            img_in,
            method=denoise_method,
        )
    if use_clahe:
        img_in = apply_clahe_pil(
            img_in,
            clip_limit=clahe_clip,
            tile_grid_size=clahe_grid,
            space=clahe_space,
        )

    
    inputs = processor(img_in, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    outputs = model(pixel_values=pixel_values)
    logits = outputs.logits  # (B,C,h,w) -> B=1

    # przeskaluj do oryginalnego rozmiaru obrazu
    target_hw = image.size[::-1]  # (W,H)->(H,W)
    up = torch.nn.functional.interpolate(logits, size=target_hw, mode="bilinear", align_corners=False)

    if return_probs:
        # probs = torch.softmax(up, dim=1).squeeze(0).cpu()  # (C,H,W)
        probs= torch.softmax(up, dim=1).squeeze(0).cpu()  # T=10.0 - wyostrzamy softmax dla lepszej wizualizacji
        pred = probs.argmax(dim=0).to(torch.int64)  # (H,W)
        return pred, probs

    pred = up.argmax(dim=1).squeeze(0).to(torch.int64).cpu()
    return pred


@torch.no_grad()
def predict_pil_image_with_thresholding(
    model: SegformerForSemanticSegmentation,
    image: "PIL.Image.Image",
    processor: SegformerImageProcessor,
    device: Optional[torch.device] = None,
    return_probs: bool = True,
    # Thresholding parameters (NEW!)
    use_threshold: bool = True,
    background_threshold: float = 0.70,    # Tło musi mieć >70% pewności
    vessel_threshold: float = 0.2,        # Naczynia muszą mieć >2% pewności
    # Denoise:
    use_denoise: bool = False,
    denoise_method: str = "nlmeans",       # 'nlmeans' | 'bilateral' | 'gaussian' | 'median'
    # CLAHE:
    use_clahe: bool = False,
    clahe_space: str = "lab",
    clahe_clip: float = 2.0,
    clahe_grid: tuple[int, int] = (8, 8),
) -> torch.Tensor:
    """
    Inferencja bezpośrednio na obiekcie PIL.Image.
    Zwraca maskę klas (H, W) jako tensor CPU (int64). Gdy return_probs=True, zwraca także mapę softmax (C,H,W).
    
    NEW FEATURE: Confidence thresholding
    - use_threshold: Włącz/wyłącz filtrowanie po pewności predykcji
    - background_threshold: Próg pewności dla tła (default 0.70 = 70%)
    - vessel_threshold: Próg pewności dla naczyń (default 0.02 = 2%)
    
    Piksele z background_prob > background_threshold ORAZ vessel_prob < vessel_threshold
    są ustawiane na tło (class 0), co eliminuje fałszywie pozytywne naczynia.
    """
    if device is None:
        device = next(model.parameters()).device

    # Preprocessing: najpierw denoise, potem CLAHE
    img_in = image
    if use_denoise:
        img_in = apply_denoise_pil(
            img_in,
            method=denoise_method,
        )
    if use_clahe:
        img_in = apply_clahe_pil(
            img_in,
            clip_limit=clahe_clip,
            tile_grid_size=clahe_grid,
            space=clahe_space,
        )

    
    inputs = processor(img_in, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    outputs = model(pixel_values=pixel_values)
    logits = outputs.logits  # (B,C,h,w) -> B=1

    # przeskaluj do oryginalnego rozmiaru obrazu
    target_hw = image.size[::-1]  # (W,H)->(H,W)
    up = torch.nn.functional.interpolate(logits, size=target_hw, mode="bilinear", align_corners=False)

    # Compute softmax probabilities
    probs = torch.softmax(up, dim=1).squeeze(0).cpu()  # (C, H, W)
    
    # Get initial prediction (argmax)
    pred = probs.argmax(dim=0).to(torch.int64)  # (H, W)
    
    # Apply confidence thresholding (NEW!)
    if use_threshold:
        background_prob = probs[0, :, :]  # Class 0 (tło)
        vessel_probs = probs[1:, :, :]    # Classes 1-26 (naczynia)
        
        # Find max vessel probability per pixel
        vessel_max_prob, _ = vessel_probs.max(dim=0)  # (H, W)
        
        # Create uncertainty mask:
        # 1. Background is very confident (>threshold) OR
        # 2. Vessel confidence is very low (<threshold)
        uncertain_mask = (background_prob > background_threshold) | (vessel_max_prob < vessel_threshold)
        
        # Set uncertain pixels to background (class 0)
        pred[uncertain_mask] = 0
        
        if return_probs:
            return pred, probs
        else:
            return pred
    
    # Original behavior (no thresholding)
    if return_probs:
        return pred, probs
    else:
        return pred

