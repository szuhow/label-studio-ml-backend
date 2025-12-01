import os
import logging
import numpy as np
import torch
import uuid

# Bezpieczny import OpenCV
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError as e:
    logging.warning(f"OpenCV import failed: {e}")
    CV2_AVAILABLE = False

from PIL import Image
import io
import base64
import requests
from typing import List, Dict, Any, Optional
from pathlib import Path

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_image_size, get_single_tag_keys

logger = logging.getLogger(__name__)

# Import funkcji z predict.py
import sys
from utils_unet import (
    create_model, 
    preprocess_single_image, 
    load_trained_model_for_inference,
    inference_single_image,
    remove_small_components_from_mask
)

# Import funkcji dla Segformer
try:
    from utils_segformer import (
        load_model_for_inference_from_checkpoint,
        predict_pil_image_with_thresholding,
        predict_pil_image,
        apply_clahe_pil,
        apply_denoise_pil
    )
    SEGFORMER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Segformer utils not available: {e}")
    SEGFORMER_AVAILABLE = False


class CoronarySegmentationModel(LabelStudioMLBase):
    """
    ML Backend dla segmentacji naczyń wieńcowych w Label Studio
    
    Wspiera:
    - Predykcje segmentacji jako RLE (Run Length Encoding)
    - Różne typy modeli (UNet, ResUNet, AttentionResUNet, etc.)
    - Automatyczne preprocessowanie obrazów
    - Konfigurowalny próg binaryzacji
    """
    
    def __init__(self, **kwargs):
        # Wydziel parametry specyficzne dla naszego modelu przed wywołaniem super().__init__()
        model_specific_params = {
            'model_path', 'model_type', 'resolution', 'threshold', 
            'min_component_size', 'endpoint', 'description',
            'smooth_mask_method', 'smooth_contour_method', 'polygon_detail_level',
            'remove_frames'
        }
        
        # Podziel kwargs na te dla klasy bazowej i te dla naszego modelu
        base_kwargs = {k: v for k, v in kwargs.items() if k not in model_specific_params}
        model_kwargs = {k: v for k, v in kwargs.items() if k in model_specific_params}
        
        # Debug: sprawdź co jest w kwargs
        logger.info(f" All kwargs keys: {list(kwargs.keys())}")
        logger.info(f" Model specific params: {model_specific_params}")
        logger.info(f" Base kwargs keys: {list(base_kwargs.keys())}")
        logger.info(f" Model kwargs keys: {list(model_kwargs.keys())}")
        if 'remove_frames' in kwargs:
            logger.info(f" remove_frames value: {kwargs['remove_frames']} (type: {type(kwargs['remove_frames'])})")
        
        # Wywołaj konstruktor klasy bazowej tylko z odpowiednimi parametrami
        super(CoronarySegmentationModel, self).__init__(**base_kwargs)
        
        # Konfiguracja modelu - pobieraj ze zmiennych środowiskowych lub kwargs
        # Inteligentne ustalanie ścieżki modelu - sprawdź która ścieżka rzeczywiście istnieje
        potential_model_paths = [
            model_kwargs.get('model_path'),  # Priorytet 1: explicit kwargs
            os.getenv('MODEL_PATH')   # Priorytet 2: zmienna środowiskowa
        ]
        
        self.model_path = None
        for path in potential_model_paths:
            if path and os.path.exists(path):
                self.model_path = path
                logger.info(f"Found existing model at: {path}")
                break
        
        # Jeśli żaden plik nie istnieje, użyj pierwszej nieNull ścieżki jako fallback
        if not self.model_path:
            for path in potential_model_paths:
                if path:
                    self.model_path = path
                    logger.warning(f"No model file found, using fallback path: {path}")
                    break
        
        self.model_type = model_kwargs.get('model_type') or os.getenv('MODEL_TYPE', 'attention_resunet')
        self.resolution = model_kwargs.get('resolution') or int(os.getenv('RESOLUTION', '320'))
        self.threshold = model_kwargs.get('threshold') or float(os.getenv('THRESHOLD', '0.5'))
        self.min_component_size = model_kwargs.get('min_component_size') or int(os.getenv('MIN_COMPONENT_SIZE', '300'))
        
        # Nowe parametry wygładzania
        self.smooth_mask_method = model_kwargs.get('smooth_mask_method') or os.getenv('SMOOTH_MASK_METHOD', 'morphology')
        self.smooth_contour_method = model_kwargs.get('smooth_contour_method') or os.getenv('SMOOTH_CONTOUR_METHOD', 'approx')
        self.polygon_detail_level = model_kwargs.get('polygon_detail_level') or os.getenv('POLYGON_DETAIL_LEVEL', 'high')  # 'low', 'medium', 'high', 'ultra'
        self.remove_frames = model_kwargs.get('remove_frames', False)  # Czy usuwać podwójne ramki
        
        # Debug info
        logger.info(f"Model configuration:")
        logger.info(f"  model_path: {self.model_path}")
        logger.info(f"  model_type: {self.model_type}")
        logger.info(f"  resolution: {self.resolution}")
        logger.info(f"  threshold: {self.threshold}")
        logger.info(f"  smooth_mask_method: {self.smooth_mask_method}")
        logger.info(f"  smooth_contour_method: {self.smooth_contour_method}")
        logger.info(f"  polygon_detail_level: {self.polygon_detail_level}")
        logger.info(f"  remove_frames: {self.remove_frames}")
        logger.info(f"  Environment MODEL_PATH: {os.getenv('MODEL_PATH', 'Not set')}")
        logger.info(f"  kwargs model_path: {model_kwargs.get('model_path', 'Not set')}")
        
        # Inicjalizuj model
        self.model = None
        self.device = None
        self._load_model()
        
        # Label Studio configuration - bezpieczne parsowanie
        self.from_name = None
        self.to_name = None
        self.value = None
        self.classes = ['coronary_artery']
        self.label_schema_classes = []  # Klasy z schema Label Studio
        
        # Spróbuj sparsować konfigurację label interface
        try:
            logger.info("Attempting to parse label interface...")
            
            if hasattr(self, 'label_interface') and self.label_interface:
                logger.info(f"Label interface type: {type(self.label_interface)}")
                
                # Użyj parsed_label_config jako główne źródło prawdy
                if hasattr(self, 'parsed_label_config'):
                    logger.info("Found parsed_label_config attribute, proceeding with parsing...")
                    try:
                        # Przeszukaj parsed_label_config bezpośrednio
                        config = self.parsed_label_config
                        logger.info(f"Parsed label config type: {type(config)}")
                        
                        # Preferuj PolygonLabels dla segmentacji naczyń wieńcowych
                        preferred_order = ['PolygonLabels', 'BrushLabels', 'RectangleLabels']
                        
                        # ZAWSZE przejdź przez config i znajdź najlepszą opcję
                        best_match = None
                        best_priority = 999
                        
                        for name, tag_info in config.items():
                            logger.info(f"Config item: {name}, type: {getattr(tag_info, 'type', 'NO_TYPE')}")
                            if hasattr(tag_info, 'type'):
                                try:
                                    priority = preferred_order.index(tag_info.type)
                                    logger.info(f"Found {tag_info.type} with priority {priority}")
                                    if priority < best_priority:
                                        best_match = (name, tag_info, tag_info.type)
                                        best_priority = priority
                                        logger.info(f"New best match: {name} ({tag_info.type})")
                                except ValueError:
                                    logger.info(f"Type {tag_info.type} not in preferred list")
                        
                        if best_match:
                            name, tag_info, tag_type = best_match
                            self.from_name = name
                            self.to_name = tag_info.to_name[0] if hasattr(tag_info, 'to_name') and tag_info.to_name else 'image'
                            
                            # Wyciągnij klasy z schema
                            if hasattr(tag_info, 'labels') and tag_info.labels:
                                self.label_schema_classes = tag_info.labels
                                logger.info(f"Found schema classes: {self.label_schema_classes}")
                            
                            logger.info(f"FINAL SELECTION from parsed_label_config: {self.from_name} ({tag_type}), {self.to_name}")
                        else:
                            logger.warning("No matching schema found in parsed_label_config")
                                
                    except Exception as e:
                        logger.warning(f"Could not parse parsed_label_config: {e}")
                        import traceback
                        logger.warning(traceback.format_exc())
                else:
                    logger.warning("No parsed_label_config attribute found")
                
                # Fallback - użyj starszej metody z obsługą błędów TYLKO jeśli nie znaleziono wcześniej
                if not self.from_name:
                    logger.info("Trying get_single_tag_keys fallback...")
                    try:
                        # Spróbuj z PolygonLabels jako pierwszym
                        for tag_type in ['PolygonLabels', 'BrushLabels', 'RectangleLabels']:
                            try:
                                self.from_name, self.to_name, self.value, self.classes = get_single_tag_keys(
                                    self.label_interface, tag_type, 'Image'
                                )
                                logger.info(f"Successfully parsed label interface with {tag_type}")
                                break
                            except Exception as inner_e:
                                logger.debug(f"Failed with {tag_type}: {inner_e}")
                                continue
                        
                        # Jeśli nic nie zadziałało, użyj domyślnych wartości
                        if not self.from_name:
                            self.from_name = 'polygon_labels'  # Preferuj polygony
                            self.to_name = 'image'
                            self.value = 'image'
                            logger.info(f"Using polygon fallback: {self.from_name}")
                                
                    except Exception as e:
                        logger.warning(f"get_single_tag_keys failed: {e}")
                        self.from_name = 'polygon_labels'  # Preferuj polygony
                        self.to_name = 'image'
                        self.value = 'image'
                        logger.info(f"Using polygon fallback after error: {self.from_name}")
            else:
                logger.warning("No label_interface found")
                
        except Exception as e:
            logger.warning(f"Error parsing label interface: {e}")
            import traceback
            traceback.print_exc()
        
        # Fallback na domyślne wartości jeśli parsowanie się nie powiodło
        if not self.from_name:
            # Dla segmentacji naczyń wieńcowych preferuj polygony zamiast prostokątów
            self.from_name = 'polygon_labels'  # Lepsze dla segmentacji konturów
            self.to_name = 'image'
            self.value = 'image'
            logger.info(f"Using polygon fallback: {self.from_name}")
        else:
            logger.info(f"Successfully parsed schema, using: {self.from_name}")
        
        # Upewnij się, że zawsze mamy klasy labelek
        if not self.label_schema_classes:
            self.label_schema_classes = ['coronary_artery']  # Domyślna klasa
            logger.info("Set default label_schema_classes: ['coronary_artery']")
        
        logger.info(f"Final schema configuration:")
        logger.info(f"  from_name: {self.from_name}")
        logger.info(f"  to_name: {self.to_name}")
        logger.info(f"  label_schema_classes: {self.label_schema_classes}")
        logger.info(f"Model: {self.model_type}, Resolution: {self.resolution}")
        logger.info(f"From: {self.from_name}, To: {self.to_name}")
    
    def get_model_version_str(self):
        """Pomocnicza metoda zwracająca model_version jako string"""
        try:
            if hasattr(self, '_model_version'):
                return str(self._model_version)
            base_version = getattr(super(), 'model_version', '1.0')
            return str(base_version)
        except:
            return "1.0"
    
    def _load_model(self):
        """Załaduj model do pamięci"""
        import torch  # Import na początku metody, żeby był dostępny w całej metodzie
        
        try:
            logger.info(f"Attempting to load model from: {self.model_path}")
            logger.info(f"File exists check: {os.path.exists(self.model_path)}")
            logger.info(f"Current working directory: {os.getcwd()}")
            
            # Sprawdź czy plik istnieje
            if os.path.exists(self.model_path):
                logger.info(f"Model file found, loading...")
                
                # Sprawdź dostępność GPU przed ładowaniem modelu
                logger.info(f"CUDA available: {torch.cuda.is_available()}")
                if torch.cuda.is_available():
                    logger.info(f"CUDA device count: {torch.cuda.device_count()}")
                    logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
                
                self.model, self.device = load_trained_model_for_inference(
                    self.model_path, 
                    self.model_type
                )
                logger.info(f"Model loaded successfully from {self.model_path}")
                logger.info(f"Device: {self.device}")
                logger.info(f"Model device: {next(self.model.parameters()).device if self.model else 'No model'}")
            else:
                logger.warning(f"Model file not found: {self.model_path}")
                
                # Sprawdź czy ścieżka jest relatywna i spróbuj z różnymi katalogami bazowymi
                if not os.path.isabs(self.model_path):
                    alternative_paths = [
                        os.path.join(os.getcwd(), self.model_path),
                        os.path.join('/app', self.model_path),
                        os.path.join(os.path.dirname(__file__), self.model_path),
                    ]
                    
                    for alt_path in alternative_paths:
                        logger.info(f"Trying alternative path: {alt_path}")
                        if os.path.exists(alt_path):
                            logger.info(f"Found model at alternative path: {alt_path}")
                            self.model_path = alt_path
                            self.model, self.device = load_trained_model_for_inference(
                                self.model_path, 
                                self.model_type
                            )
                            logger.info(f"Model loaded successfully from {self.model_path}")
                            logger.info(f"Device: {self.device}")
                            return
                
                # Jeśli nadal nie znaleziono, stwórz dummy model
                logger.warning(f"Model file not found at any location. Creating dummy model for testing.")
                self.model = create_model(self.model_type, n_class=1)
                self.device = torch.device('cpu')
                logger.info("Created dummy model for testing")
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback na dummy model
            logger.info("Falling back to dummy model due to loading error")
            try:
                self.model = create_model(self.model_type, n_class=1)
                self.device = torch.device('cpu')
                logger.info("Dummy model created successfully")
            except Exception as e2:
                logger.error(f"Even dummy model creation failed: {e2}")
                self.model = None
                self.device = torch.device('cpu')
    
    def _download_image(self, url: str) -> Image.Image:
        """Pobierz obraz z URL"""
        try:
            if url.startswith('data:'):
                # Base64 encoded image
                header, data = url.split(',', 1)
                image_data = base64.b64decode(data)
                return Image.open(io.BytesIO(image_data))
            else:
                # URL do obrazu
                response = requests.get(url)
                return Image.open(io.BytesIO(response.content))
        except Exception as e:
            logger.error(f"Error downloading image from {url}: {e}")
            raise
    
    def _get_image_from_task(self, task: Dict) -> Image.Image:
        """Pobierz obraz z zadania Label Studio"""
        # Znajdź klucz obrazu w danych zadania
        image_key = self.value if self.value else 'image'
        
        # Sprawdź różne możliwe klucze
        possible_keys = [image_key, 'image', 'data', 'url']
        image_url = None
        
        for key in possible_keys:
            if key in task.get('data', {}):
                image_url = task['data'][key]
                break
        
        if not image_url:
            # Fallback - weź pierwszy klucz z task['data']
            data_keys = list(task.get('data', {}).keys())
            if data_keys:
                image_url = task['data'][data_keys[0]]
                logger.info(f"Using fallback image key: {data_keys[0]}")
        
        if not image_url:
            raise ValueError("No image URL found in task data")
        
        # Sprawdź czy to lokalna ścieżka czy URL
        if image_url.startswith('http') or image_url.startswith('data:'):
            return self._download_image(image_url)
        else:
            # Lokalna ścieżka - względem LABEL_STUDIO_URL
            if hasattr(self, 'label_studio_url') and self.label_studio_url:
                full_url = f"{self.label_studio_url.rstrip('/')}/{image_url.lstrip('/')}"
                return self._download_image(full_url)
            else:
                # Bezpośrednia ścieżka
                return Image.open(image_url)
    
    def _remove_small_components(self, mask: np.ndarray, min_size: int = 300) -> np.ndarray:
        """Usuwa małe komponenty z binarnej maski używając OpenCV"""
        try:
            if CV2_AVAILABLE:
                # Konwertuj na uint8
                mask_uint8 = (mask * 255).astype(np.uint8)
                
                # Użyj 4-połączeniowej analizy dla lepszej separacji komponentów
                # To pomoże rozdzielić naczynia które są blisko siebie ale nie połączone
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                    mask_uint8, connectivity=4
                )
                
                # Stwórz nową pustą maskę
                mask_cleaned = np.zeros_like(mask, dtype=np.uint8)
                
                # Zostaw tylko duże komponenty
                for i in range(1, num_labels):  # pomiń tło (i=0)
                    if stats[i, cv2.CC_STAT_AREA] >= min_size:
                        mask_cleaned[labels == i] = 1
                
                logger.info(f"Component removal: {num_labels-1} components found, kept {sum(1 for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] >= min_size)}")
                return mask_cleaned
            else:
                # Brak OpenCV - zwróć oryginalną maskę
                logger.warning("OpenCV not available, skipping component removal")
                return mask.astype(np.uint8)
                
        except Exception as e:
            logger.warning(f"Error in component removal: {e}, returning original mask")
            return mask.astype(np.uint8)
    
    def _smooth_mask(self, mask: np.ndarray, method='gaussian', **kwargs) -> np.ndarray:
        """
        Wygładza maskę segmentacji dla uzyskania gładszych konturów
        
        Args:
            mask: Maska binarna (0-1)
            method: Metoda wygładzania ('gaussian', 'morphology', 'bilateral', 'combined')
            **kwargs: Dodatkowe parametry dla metod
        
        Returns:
            Wygładzona maska
        """
        if not CV2_AVAILABLE:
            return mask
            
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        if method == 'gaussian':
            kernel_size = kwargs.get('kernel_size', 5)
            sigma = kwargs.get('sigma', 1.0)
            # Gaussian blur
            smoothed = cv2.GaussianBlur(mask_uint8, (kernel_size, kernel_size), sigma)
            # Ponowna binaryzacja
            _, smoothed = cv2.threshold(smoothed, 127, 255, cv2.THRESH_BINARY)
            
        elif method == 'morphology':
            # Operacje morfologiczne - zamknięcie + otwarcie
            kernel_size = kwargs.get('kernel_size', 3)
            iterations = kwargs.get('iterations', 1)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            # Zamknięcie (closing) - wypełnia małe dziury
            smoothed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=iterations)
            # Otwarcie (opening) - usuwa małe szumy
            smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, kernel, iterations=iterations)
            
        elif method == 'bilateral':
            # Bilateral filter - zachowuje krawędzie ale wygładza
            d = kwargs.get('d', 9)
            sigma_color = kwargs.get('sigma_color', 75)
            sigma_space = kwargs.get('sigma_space', 75)
            
            smoothed = cv2.bilateralFilter(mask_uint8, d, sigma_color, sigma_space)
            _, smoothed = cv2.threshold(smoothed, 127, 255, cv2.THRESH_BINARY)
            
        elif method == 'combined':
            # Kombinacja metod dla najlepszego efektu
            kernel_size = kwargs.get('kernel_size', 3)
            
            # 1. Morfologia do wypełnienia dziur
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            smoothed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=1)
            
            # 2. Gaussian blur dla wygładzenia
            smoothed = cv2.GaussianBlur(smoothed, (5, 5), 1.0)
            
            # 3. Ponowna binaryzacja
            _, smoothed = cv2.threshold(smoothed, 127, 255, cv2.THRESH_BINARY)
            
        else:
            smoothed = mask_uint8
            
        return (smoothed / 255.0).astype(np.float32)

    def _smooth_contour(self, contour: np.ndarray, method='approx', **kwargs) -> np.ndarray:
        """
        Wygładza pojedynczy kontur
        
        Args:
            contour: Kontur z OpenCV
            method: Metoda wygładzania ('approx', 'gaussian')
            **kwargs: Dodatkowe parametry
            
        Returns:
            Wygładzony kontur
        """
        if not CV2_AVAILABLE or len(contour) < 3:
            return contour
            
        if method == 'approx':
            # Aproksymacja Douglas-Peucker z dostrojoną dokładnością
            epsilon_factor = kwargs.get('epsilon_factor', 0.001)  # Mniejsza wartość = więcej szczegółów
            epsilon = epsilon_factor * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            return approx
            
        elif method == 'gaussian':
            # Proste wygładzenie przez uśrednienie punktów sąsiadujących
            window_size = kwargs.get('window_size', 3)
            if window_size < 3:
                return contour
                
            contour_2d = contour.reshape(-1, 2).astype(np.float32)
            n_points = len(contour_2d)
            smoothed_points = np.zeros_like(contour_2d)
            
            for i in range(n_points):
                # Zbierz punkty w oknie
                window_points = []
                for j in range(-window_size//2, window_size//2 + 1):
                    idx = (i + j) % n_points  # Circular indexing
                    window_points.append(contour_2d[idx])
                
                # Uśrednij
                smoothed_points[i] = np.mean(window_points, axis=0)
            
            return smoothed_points.astype(np.int32).reshape(-1, 1, 2)
            
        return contour
    
    def _mask_to_polygons(self, mask: np.ndarray, original_width: int, original_height: int, min_area: int = 100) -> List[List[List[float]]]:
        """
        Konwertuj maskę binarną na listę polygonów z zaawansowanym wygładzaniem
        
        Args:
            mask: Binarna maska (0 i 1)
            original_width: Szerokość oryginalnego obrazu
            original_height: Wysokość oryginalnego obrazu
            min_area: Minimalny obszar konturu do uwzględnienia
            
        Returns:
            Lista polygonów, gdzie każdy polygon to lista punktów [x, y] w procentach
        """
        try:
            polygons = []
            
            # Wygładź maskę przed znajdowaniem konturów
            if self.smooth_mask_method != 'none':
                mask = self._smooth_mask(mask, method=self.smooth_mask_method, kernel_size=3)
            

            
            if CV2_AVAILABLE:
                # Użyj OpenCV do znajdowania konturów
                mask_uint8 = (mask * 255).astype(np.uint8)
                
                # Użyj OpenCV do znajdowania konturów
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    # Sprawdź czy kontur jest wystarczająco duży
                    area = cv2.contourArea(contour)
                    if area < min_area:
                        continue
                    
                    # Parametry aproksymacji na podstawie poziomu szczegółowości
                    if self.polygon_detail_level == 'ultra':
                        # Bardzo wysokie szczegóły - minimalna aproksymacja
                        epsilon_factor = 0.00005  # Jeszcze mniejsza aproksymacja
                        max_points = 200  # Więcej punktów
                        min_points = 30
                        use_full_contour = True  # Użyj pełnego konturu
                    elif self.polygon_detail_level == 'maximum':
                        # Maksymalne szczegóły - prawie bez aproksymacji
                        epsilon_factor = 0.00001  # Minimalna aproksymacja
                        max_points = 500  # Bardzo dużo punktów
                        min_points = 50
                        use_full_contour = True
                    elif self.polygon_detail_level == 'high':
                        # Wysokie szczegóły
                        epsilon_factor = 0.0002
                        max_points = 100
                        min_points = 20
                        use_full_contour = True
                    elif self.polygon_detail_level == 'medium':
                        # Średnie szczegóły
                        epsilon_factor = 0.0005
                        max_points = 50
                        min_points = 15
                        use_full_contour = False
                    else:  # 'low'
                        # Niskie szczegóły
                        epsilon_factor = 0.001
                        max_points = 30
                        min_points = 8
                        use_full_contour = False
                    
                    # Jeśli chcemy wysokie szczegóły, zacznij od pełnego konturu
                    if use_full_contour:
                        # Pobierz pełny kontur bez aproksymacji
                        full_contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                        
                        # Znajdź kontur odpowiadający obecnemu obszarowi
                        for full_contour in full_contours:
                            if abs(cv2.contourArea(full_contour) - area) < area * 0.1:  # 10% tolerancja
                                contour = full_contour
                                break
                    
                    # Wygładź kontur jeśli wymagane
                    if self.smooth_contour_method != 'none':
                        if self.smooth_contour_method == 'approx':
                            contour = self._smooth_contour(contour, method='approx', epsilon_factor=epsilon_factor)
                        elif self.smooth_contour_method == 'gaussian':
                            contour = self._smooth_contour(contour, method='gaussian', window_size=3)
                    
                    # Równomierne próbkowanie jeśli za dużo punktów
                    if len(contour) > max_points:
                        # Równomiernie próbkuj kontury zachowując naturalne kształty
                        step = len(contour) // max_points
                        contour = contour[::step]
                    
                    # Jeśli nadal za mało punktów, dodaj więcej z oryginalnego konturu
                    if len(contour) < min_points and use_full_contour:
                        # Próbkuj gęściej z pełnego konturu
                        step = max(1, len(contour) // min_points)
                        contour = contour[::step]
                    
                    # Debug: zapisz statystyki punktów
                    logger.debug(f"Polygon stats - Points: {len(contour)}, Detail level: {self.polygon_detail_level}, Area: {area:.0f}, Max points: {max_points}")
                    
                    # Konwertuj na listę punktów w procentach
                    polygon_points = []
                    for point in contour:
                        try:
                            # Bezpieczne rozpakowywanie punktów OpenCV
                            if len(point.shape) == 3 and point.shape[0] == 1:  # Format (1, 2) z OpenCV
                                x, y = point[0]
                            elif len(point.shape) == 2 and point.shape[0] == 1:  # Format (1, 2) alternatywny
                                x, y = point[0]
                            elif len(point.shape) == 1 and len(point) == 2:  # Format (2,)
                                x, y = point
                            elif hasattr(point, '__len__') and len(point) >= 2:  # Lista lub tuple
                                x, y = point[0], point[1]
                            else:
                                logger.warning(f"Unexpected point format: {point}, shape: {getattr(point, 'shape', 'no shape')}")
                                continue
                                
                            x_pct = float((x / original_width) * 100)
                            y_pct = float((y / original_height) * 100)
                            polygon_points.append([x_pct, y_pct])
                            
                            # Debug: sprawdź pierwszy punkt
                            if len(polygon_points) == 1:
                                logger.info(f"First polygon point: ({x}, {y}) -> ({x_pct:.2f}%, {y_pct:.2f}%)")
                        except Exception as point_error:
                            logger.warning(f"Error processing point {point}: {point_error}")
                            continue
                    
                    # Dodaj polygon tylko jeśli ma przynajmniej 3 punkty
                    if len(polygon_points) >= 3:
                        polygons.append(polygon_points)
                        
            else:
                # Fallback - bounding box
                logger.warning("OpenCV not available, using bounding box fallback")
                y_coords, x_coords = np.where(mask > 0)
                if len(x_coords) > 0 and len(y_coords) > 0:
                    x_min, x_max = float(x_coords.min()), float(x_coords.max())
                    y_min, y_max = float(y_coords.min()), float(y_coords.max())
                    
                    x_min_pct = float((x_min / original_width) * 100)
                    y_min_pct = float((y_min / original_height) * 100)
                    x_max_pct = float((x_max / original_width) * 100)
                    y_max_pct = float((y_max / original_height) * 100)
                    
                    # Prostokątny polygon
                    polygon_points = [
                        [x_min_pct, y_min_pct],
                        [x_max_pct, y_min_pct],
                        [x_max_pct, y_max_pct],
                        [x_min_pct, y_max_pct]
                    ]
                    polygons.append(polygon_points)
            
            return polygons
            
        except Exception as e:
            logger.error(f"Error converting mask to polygons: {e}")
            return []
    
    def _mask_to_rle(self, mask: np.ndarray) -> List[int]:
        """Konwertuj maskę do RLE (Run Length Encoding)"""
        flat_mask = mask.flatten()
        diff = np.diff(np.concatenate(([0], flat_mask, [0])))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        rle = []
        for start, end in zip(starts, ends):
            rle.extend([int(start), int(end - start)])
        return rle
    
    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs):
        """
        Główna funkcja predykcji dla Label Studio
        
        Args:
            tasks: Lista zadań z Label Studio
            context: Kontekst (dla interactive labeling)
            
        Returns:
            Lista predykcji w formacie Label Studio
        """
        logger.info(f"Received {len(tasks)} tasks for prediction")
        
        print(f'''\
        Run prediction on {tasks}
        Received context: {context}
        Project ID: {getattr(self, 'project_id', 'Unknown')}
        Label config: {getattr(self, 'label_config', 'Unknown')}
        Parsed JSON Label config: {getattr(self, 'parsed_label_config', 'Unknown')}
        Extra params: {getattr(self, 'extra_params', {})}''')
        
        if self.model is None:
            logger.error("Model not loaded")
            return [{'result': [], 'score': 0.0, 'model_version': self.get_model_version_str()} for _ in tasks]
        
        predictions = []
        
        for task in tasks:
            try:
                # Pobierz obraz
                image = self._get_image_from_task(task)
                logger.info(f"Processing image: {image.size}")
                
                # Preprocessing
                image_tensor, original_image = preprocess_single_image(
                    image, 
                    resolution=self.resolution,
                    apply_preprocessing=True,
                    remove_frames=self.remove_frames
                )
                
                # Inferencja
                prediction_soft, prediction_binary, confidence = inference_single_image(
                    self.model, 
                    image_tensor, 
                    self.device, 
                    threshold=self.threshold
                )
                
                # Postprocessing - usuń małe komponenty
                prediction_binary_np = prediction_binary.cpu().squeeze().numpy()
                prediction_clean = self._remove_small_components(
                    prediction_binary_np, 
                    min_size=self.min_component_size
                )
                
                # Przeskaluj maskę do oryginalnego rozmiaru obrazu
                original_height, original_width = image.size[1], image.size[0]
                logger.info(f"Scaling mask from {prediction_clean.shape} to ({original_width}, {original_height})")
                
                if CV2_AVAILABLE:
                    mask_resized = cv2.resize(
                        prediction_clean.astype(np.uint8), 
                        (original_width, original_height), 
                        interpolation=cv2.INTER_NEAREST
                    )
                else:
                    # Fallback bez OpenCV
                    mask_pil = Image.fromarray((prediction_clean * 255).astype(np.uint8))
                    mask_resized = np.array(mask_pil.resize(
                        (original_width, original_height), 
                        Image.NEAREST
                    )) > 128
                
                # Konwertuj do polygonów lub RLE na podstawie typu schema
                polygons = self._mask_to_polygons(mask_resized, original_width, original_height)
                
                # Wybierz klasę do użycia - pierwsza ze schema lub fallback
                prediction_class = self.label_schema_classes[0] if (self.label_schema_classes and len(self.label_schema_classes) > 0) else 'coronary_artery'
                
                # Upewnij się, że klasa nie jest pusta
                if not prediction_class or prediction_class.strip() == '':
                    prediction_class = 'coronary_artery'
                
                logger.info(f"Using prediction class: {prediction_class}")
                
                # Przygotuj wynik w formacie Label Studio
                if len(polygons) > 0:  # Jeśli znaleziono jakieś obiekty
                    results = []
                    
                    # Utwórz wynik dla każdego znalezionego polygonu
                    for polygon_points in polygons:
                        
                        # Określ typ wyniku na podstawie from_name
                        if self.from_name and 'rectangle' in self.from_name.lower():
                            # Dla RectangleLabels - konwertuj polygon na bounding box
                            if len(polygon_points) > 0:
                                x_coords = [p[0] for p in polygon_points]
                                y_coords = [p[1] for p in polygon_points]
                                
                                x_min_pct = min(x_coords)
                                y_min_pct = min(y_coords)
                                x_max_pct = max(x_coords)
                                y_max_pct = max(y_coords)
                                
                                result_type = 'rectanglelabels'
                                result_value = {
                                    'x': float(x_min_pct),
                                    'y': float(y_min_pct),
                                    'width': float(x_max_pct - x_min_pct),
                                    'height': float(y_max_pct - y_min_pct),
                                    'rectanglelabels': [prediction_class]
                                }
                        elif self.from_name and 'brush' in self.from_name.lower():
                            # Dla BrushLabels - użyj RLE tylko dla pierwszego polygonu
                            if polygon_points == polygons[0]:  # Tylko dla pierwszego
                                rle = self._mask_to_rle(mask_resized)
                                rle = [int(x) for x in rle]  # Konwertuj na standardowe int
                                
                                result_type = 'brushlabels'
                                result_value = {
                                    'format': 'rle',
                                    'rle': rle,
                                    'brushlabels': [prediction_class]
                                }
                            else:
                                continue  # Pomiń pozostałe polygony dla brush
                        else:
                            # Domyślnie lub gdy from_name zawiera 'polygon' - PolygonLabels z rzeczywistymi konturami
                            result_type = 'polygonlabels'
                            result_value = {
                                'points': polygon_points,
                                'polygonlabels': [prediction_class]
                            }
                        
                        result = {
                            'from_name': self.from_name,
                            'to_name': self.to_name,
                            'type': result_type,
                            'value': result_value
                        }
                        results.append(result)
                    
                    prediction = {
                        'result': results,
                        'score': float(confidence),
                        'model_version': self.get_model_version_str()
                    }
                else:
                    # Brak wykrytych obiektów
                    prediction = {
                        'result': [],
                        'score': float(confidence),
                        'model_version': self.get_model_version_str()
                    }
                
                predictions.append(prediction)
                
                logger.info(f"Prediction completed - confidence: {confidence:.4f}, "
                           f"positive pixels: {mask_resized.sum()}")
                
            except Exception as e:
                logger.error(f"Error in prediction: {e}")
                import traceback
                traceback.print_exc()
                
                # Zwróć pustą predykcję w przypadku błędu
                predictions.append({
                    'result': [],
                    'score': 0.0,
                    'model_version': self.get_model_version_str()
                })
        
        return predictions
    
    def fit(self, event, data, **kwargs):
        """Placeholder dla trenowania modelu"""
        logger.info(f"Fit called with event: {event}")
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Zwróć informacje o modelu"""
        return {
            'model_type': self.model_type,
            'model_path': self.model_path,
            'resolution': self.resolution,
            'threshold': self.threshold,
            'device': str(self.device),
            'model_loaded': self.model is not None,
            'classes': ['coronary_artery']
        }


class SegFormerSegmentationModel(LabelStudioMLBase):
    """
    ML Backend dla segmentacji naczyń wieńcowych w Label Studio używający Segformer
    
    Wspiera:
    - Predykcje segmentacji jako RLE (Run Length Encoding)
    - Polygony i maski dla Label Studio
    - Preprocessing: CLAHE, denoising
    - Thresholding po pewności predykcji
    """
    
    def __init__(self, **kwargs):
        # Wydziel parametry specyficzne dla Segformer
        model_specific_params = {
            'model_path', 'model_type', 'model_size', 'resolution', 'num_classes',
            'threshold', 'background_threshold', 'vessel_threshold',
            'min_component_size', 'endpoint', 'description',
            'smooth_mask_method', 'smooth_contour_method', 'polygon_detail_level',
            'use_clahe', 'clahe_space', 'clahe_clip', 'clahe_grid',
            'use_denoise', 'denoise_method', 'remove_frames'
        }
        
        base_kwargs = {k: v for k, v in kwargs.items() if k not in model_specific_params}
        model_kwargs = {k: v for k, v in kwargs.items() if k in model_specific_params}
        
        super(SegFormerSegmentationModel, self).__init__(**base_kwargs)
        
        # Konfiguracja modelu
        potential_model_paths = [
            model_kwargs.get('model_path'),
            os.getenv('MODEL_PATH')
        ]
        
        self.model_path = None
        for path in potential_model_paths:
            if path and os.path.exists(path):
                self.model_path = path
                logger.info(f"Found existing model at: {path}")
                break
        
        if not self.model_path:
            for path in potential_model_paths:
                if path:
                    self.model_path = path
                    logger.warning(f"No model file found, using fallback path: {path}")
                    break
        
        self.model_type = model_kwargs.get('model_type') or os.getenv('MODEL_TYPE', 'segformer')
        self.model_size = model_kwargs.get('model_size') or os.getenv('MODEL_SIZE', 'b4')
        self.resolution = model_kwargs.get('resolution') or int(os.getenv('RESOLUTION', '512'))
        self.num_classes = model_kwargs.get('num_classes') or int(os.getenv('NUM_CLASSES', '2'))
        self.threshold = model_kwargs.get('threshold') or float(os.getenv('THRESHOLD', '0.5'))
        self.background_threshold = model_kwargs.get('background_threshold') or float(os.getenv('BACKGROUND_THRESHOLD', '0.70'))
        self.vessel_threshold = model_kwargs.get('vessel_threshold') or float(os.getenv('VESSEL_THRESHOLD', '0.2'))
        self.min_component_size = model_kwargs.get('min_component_size') or int(os.getenv('MIN_COMPONENT_SIZE', '300'))
        
        # Preprocessing
        self.use_clahe = model_kwargs.get('use_clahe', False) or (os.getenv('USE_CLAHE', 'false').lower() == 'true')
        self.clahe_space = model_kwargs.get('clahe_space') or os.getenv('CLAHE_SPACE', 'lab')
        self.clahe_clip = model_kwargs.get('clahe_clip') or float(os.getenv('CLAHE_CLIP', '2.0'))
        clahe_grid_raw = model_kwargs.get('clahe_grid') or os.getenv('CLAHE_GRID', '8,8')
        # Upewnij się, że clahe_grid jest tuple z dokładnie 2 elementami
        if isinstance(clahe_grid_raw, str):
            parts = clahe_grid_raw.split(',')
            self.clahe_grid = tuple(map(int, parts[:2]))  # Weź tylko pierwsze 2 elementy
        elif isinstance(clahe_grid_raw, (list, tuple)):
            self.clahe_grid = tuple(map(int, clahe_grid_raw[:2]))  # Weź tylko pierwsze 2 elementy
        else:
            self.clahe_grid = (8, 8)  # Domyślna wartość
        
        self.use_denoise = model_kwargs.get('use_denoise', False) or (os.getenv('USE_DENOISE', 'false').lower() == 'true')
        self.denoise_method = model_kwargs.get('denoise_method') or os.getenv('DENOISE_METHOD', 'nlmeans')
        
        # Postprocessing
        self.smooth_mask_method = model_kwargs.get('smooth_mask_method') or os.getenv('SMOOTH_MASK_METHOD', 'morphology')
        self.smooth_contour_method = model_kwargs.get('smooth_contour_method') or os.getenv('SMOOTH_CONTOUR_METHOD', 'approx')
        self.polygon_detail_level = model_kwargs.get('polygon_detail_level') or os.getenv('POLYGON_DETAIL_LEVEL', 'high')
        self.remove_frames = model_kwargs.get('remove_frames', False)
        
        logger.info(f"SegFormer Model configuration:")
        logger.info(f"  model_path: {self.model_path}")
        logger.info(f"  model_size: {self.model_size}")
        logger.info(f"  resolution: {self.resolution}")
        logger.info(f"  num_classes: {self.num_classes}")
        logger.info(f"  background_threshold: {self.background_threshold}")
        logger.info(f"  vessel_threshold: {self.vessel_threshold}")
        logger.info(f"  use_clahe: {self.use_clahe}")
        logger.info(f"  use_denoise: {self.use_denoise}")
        
        # Inicjalizuj model
        self.model = None
        self.processor = None
        self.device = None
        self._load_model()
        
        # Label Studio configuration
        self.from_name = None
        self.to_name = None
        self.value = None
        self.classes = ['coronary_artery']
        self.label_schema_classes = []
        
        try:
            if hasattr(self, 'parsed_label_config') and self.parsed_label_config:
                config = self.parsed_label_config
                # Dla Segformer preferuj BrushLabels (używamy RLE)
                preferred_order = ['BrushLabels', 'PolygonLabels', 'RectangleLabels']
                best_match = None
                best_priority = 999
                
                for name, tag_info in config.items():
                    if hasattr(tag_info, 'type'):
                        try:
                            priority = preferred_order.index(tag_info.type)
                            if priority < best_priority:
                                best_match = (name, tag_info, tag_info.type)
                                best_priority = priority
                        except ValueError:
                            pass
                
                if best_match:
                    name, tag_info, tag_type = best_match
                    self.from_name = name
                    self.to_name = tag_info.to_name[0] if hasattr(tag_info, 'to_name') and tag_info.to_name else 'image'
                    
                    # Spróbuj różne sposoby pobrania klas
                    if hasattr(tag_info, 'labels') and tag_info.labels:
                        self.label_schema_classes = tag_info.labels
                    elif hasattr(tag_info, 'label') and tag_info.label:
                        # Może być pojedyncza klasa
                        self.label_schema_classes = [tag_info.label] if isinstance(tag_info.label, str) else tag_info.label
                    elif hasattr(tag_info, 'children') and tag_info.children:
                        # Spróbuj wyciągnąć z children (Label elements)
                        labels = []
                        for child in tag_info.children:
                            if hasattr(child, 'value'):
                                labels.append(child.value)
                        if labels:
                            self.label_schema_classes = labels
                    
                    logger.info(f"Segformer: Found {len(self.label_schema_classes)} labels: {self.label_schema_classes}")
                    if not self.label_schema_classes:
                        logger.warning(f"Segformer: No labels found in tag_info. Available attributes: {dir(tag_info)}")
                        if hasattr(tag_info, 'children'):
                            logger.warning(f"Segformer: tag_info.children: {tag_info.children}")
                    logger.info(f"Segformer: Selected {tag_type} with from_name='{self.from_name}', to_name='{self.to_name}'")
        except Exception as e:
            logger.warning(f"Error parsing label interface: {e}")
        
        if not self.from_name:
            # Fallback dla BrushLabels (używamy RLE)
            self.from_name = 'brush_labels'
            self.to_name = 'image'
            logger.warning(f"Segformer: Using fallback from_name='{self.from_name}'")
            self.value = 'image'
        
        if not self.label_schema_classes:
            self.label_schema_classes = ['coronary_artery']
    
    def get_model_version_str(self):
        try:
            if hasattr(self, '_model_version'):
                return str(self._model_version)
            return "1.0"
        except:
            return "1.0"
    
    def _load_model(self):
        """Załaduj model Segformer"""
        if not SEGFORMER_AVAILABLE:
            logger.error("Segformer utils not available")
            return
        
        try:
            logger.info(f"Loading Segformer model from: {self.model_path}")
            
            if not self.model_path or not os.path.exists(self.model_path):
                logger.warning(f"Model file not found: {self.model_path}")
                return
            
            import torch
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")
            
            self.model, self.processor = load_model_for_inference_from_checkpoint(
                checkpoint_path=self.model_path,
                model_size=self.model_size,
                image_size=self.resolution,
                num_classes=self.num_classes,
                device=self.device
            )
            
            logger.info(f"Segformer model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading Segformer model: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
            self.processor = None
    
    def _download_image(self, url: str) -> Image.Image:
        """Pobierz obraz z URL (obsługuje lokalne pliki Label Studio i CloudFront URLs)"""
        try:
            if url.startswith('data:'):
                header, data = url.split(',', 1)
                image_data = base64.b64decode(data)
                return Image.open(io.BytesIO(image_data))
            else:
                # CloudFront i inne CDN mogą wymagać dodatkowych nagłówków
                headers = {
                    'User-Agent': 'Mozilla/5.0 (compatible; LabelStudioML/1.0)',
                    'Accept': 'image/*,*/*'
                }
                
                # Dla CloudFront może być potrzebny redirect handling
                response = requests.get(url, timeout=30, stream=True, headers=headers, allow_redirects=True)
                response.raise_for_status()
                
                # Sprawdź czy content nie jest pusty
                if len(response.content) == 0:
                    raise ValueError(f"Empty response from {url}")
                
                # Sprawdź magic bytes obrazu (bardziej niezawodne niż content-type)
                image_data = response.content
                is_image = False
                
                # Sprawdź magic bytes dla popularnych formatów
                if len(image_data) >= 4:
                    magic_bytes = image_data[:4]
                    # PNG: 89 50 4E 47
                    # JPEG: FF D8 FF E0/FF D8 FF E1/FF D8 FF DB
                    # GIF: 47 49 46 38
                    # BMP: 42 4D
                    if (magic_bytes[:3] == b'\x89PN' or  # PNG
                        magic_bytes[:2] == b'\xFF\xD8' or  # JPEG
                        magic_bytes[:4] == b'GIF8' or  # GIF
                        magic_bytes[:2] == b'BM'):  # BMP
                        is_image = True
                
                # Sprawdź content-type jako fallback
                content_type = response.headers.get('content-type', '').lower()
                if not is_image and ('image/' in content_type):
                    is_image = True
                
                # Jeśli nie wygląda na obraz, sprawdź czy to HTML/tekst
                if not is_image:
                    # Sprawdź czy to HTML (często CloudFront zwraca HTML przy błędach)
                    if (b'<html' in image_data[:200].lower() or 
                        b'<!doctype' in image_data[:200].lower() or
                        'text/html' in content_type or 
                        'text/plain' in content_type):
                        content_preview = image_data[:500].decode('utf-8', errors='ignore')
                        logger.error(f"Received HTML/text instead of image from {url[:150]}...")
                        logger.error(f"Content-Type: {content_type}, Content preview: {content_preview[:200]}")
                        raise ValueError(f"Server returned HTML/text instead of image. URL: {url[:150]}")
                
                # Spróbuj otworzyć obraz
                try:
                    image = Image.open(io.BytesIO(image_data))
                    image.load()  # Wymusza załadowanie danych
                    return image
                except Exception as img_error:
                    logger.error(f"Failed to parse image from {url[:150]}... Error: {img_error}")
                    logger.error(f"Content-Type: {content_type}, Content length: {len(image_data)}")
                    # Pokaż magic bytes
                    if len(image_data) >= 16:
                        magic_hex = image_data[:16].hex()
                        logger.error(f"Magic bytes (hex): {magic_hex}")
                    raise
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP error downloading image from {url[:150]}...: {e}")
            raise
        except Exception as e:
            logger.error(f"Error downloading image from {url[:150]}...: {e}")
            raise
    
    def _get_image_from_task(self, task: Dict) -> Image.Image:
        """Pobierz obraz z zadania Label Studio (obsługuje lokalne pliki i CloudFront URLs)"""
        # DEBUG: Zahardkodowany obraz testowy
        # Sprawdź najpierw w kontenerze Docker, potem na hoście
        debug_image_paths = [
            '/app/image/1.png',  # W kontenerze Docker
            '/home/rafal/Dokumenty/ivessystem/coronary/label-studio-ml-backend/image/1.png'  # Na hoście
        ]
        for debug_image_path in debug_image_paths:
            if os.path.exists(debug_image_path):
                logger.warning(f"DEBUG MODE: Using hardcoded test image: {debug_image_path}")
                img = Image.open(debug_image_path)
                logger.info(f"DEBUG IMAGE LOADED: path={debug_image_path}, size={img.size}, mode={img.mode}")
                # Loguj statystyki obrazu
                import numpy as np
                img_array = np.array(img)
                logger.info(f"DEBUG IMAGE STATS: shape={img_array.shape}, dtype={img_array.dtype}, "
                           f"min={img_array.min()}, max={img_array.max()}, mean={img_array.mean():.2f}")
                return img
        
        image_key = self.value if self.value else 'image'
        possible_keys = [image_key, 'image', 'data', 'url']
        image_url = None
        
        for key in possible_keys:
            if key in task.get('data', {}):
                image_url = task['data'][key]
                break
        
        if not image_url:
            data_keys = list(task.get('data', {}).keys())
            if data_keys:
                image_url = task['data'][data_keys[0]]
                logger.info(f"Using fallback image key: {data_keys[0]}")
        
        if not image_url:
            raise ValueError("No image URL found in task data")
        
        logger.debug(f"Image URL from task: {image_url[:200]}")
        
        # Sprawdź czy to URL HTTP/HTTPS (w tym CloudFront) lub base64
        if image_url.startswith('http://') or image_url.startswith('https://') or image_url.startswith('data:'):
            # Pełny URL (CloudFront, S3, itp.) lub base64
            return self._download_image(image_url)
        else:
            # Lokalna ścieżka - może być w Label Studio lub na dysku
            # Najpierw sprawdź czy plik istnieje lokalnie
            if os.path.exists(image_url):
                logger.debug(f"Found local file: {image_url}")
                return Image.open(image_url)
            
            # Jeśli nie istnieje lokalnie, spróbuj pobrać przez Label Studio API
            label_studio_url = None
            
            # Sprawdź czy mamy label_studio_url z setup request
            if hasattr(self, 'label_studio_url') and self.label_studio_url:
                label_studio_url = self.label_studio_url
            else:
                # Fallback do zmiennej środowiskowej
                label_studio_url = os.getenv('LABEL_STUDIO_URL', 'http://localhost:8080')
            
            if label_studio_url:
                # Label Studio często używa /data/upload/... jako ścieżki
                # Usuń duplikaty slashes i zbuduj pełny URL
                base_url = label_studio_url.rstrip('/')
                path = image_url.lstrip('/')
                full_url = f"{base_url}/{path}"
                
                logger.info(f"Local file not found, trying Label Studio URL: {full_url}")
                try:
                    return self._download_image(full_url)
                except Exception as e:
                    logger.warning(f"Failed to download from Label Studio URL {full_url}: {e}")
                    # Jeśli to nie zadziałało, spróbuj jeszcze raz z /data/upload jeśli ścieżka nie zaczyna się od tego
                    if not path.startswith('data/upload') and not path.startswith('/data/upload'):
                        alt_url = f"{base_url}/data/upload/{path}"
                        logger.info(f"Trying alternative Label Studio URL: {alt_url}")
                        try:
                            return self._download_image(alt_url)
                        except Exception as alt_e:
                            logger.error(f"Failed to download from alternative URL {alt_url}: {alt_e}")
                            raise FileNotFoundError(f"Image not found locally and could not be downloaded from Label Studio: {image_url}")
                    else:
                        raise FileNotFoundError(f"Image not found locally and could not be downloaded from Label Studio: {image_url}")
            else:
                raise FileNotFoundError(f"Image file not found: {image_url} and LABEL_STUDIO_URL not set")
    
    def _remove_small_components(self, mask: np.ndarray, min_size: int = 300) -> np.ndarray:
        """Usuwa małe komponenty z binarnej maski"""
        try:
            if CV2_AVAILABLE:
                mask_uint8 = (mask * 255).astype(np.uint8)
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=4)
                mask_cleaned = np.zeros_like(mask, dtype=np.uint8)
                
                for i in range(1, num_labels):
                    area = stats[i, cv2.CC_STAT_AREA]
                    if area >= min_size:
                        mask_cleaned[labels == i] = 1
                
                return mask_cleaned
            else:
                return mask
        except Exception as e:
            logger.warning(f"Error removing small components: {e}")
            return mask
    
    def _smooth_mask(self, mask: np.ndarray, method: str = 'morphology', kernel_size: int = 3) -> np.ndarray:
        """Wygładź maskę"""
        if not CV2_AVAILABLE or method == 'none':
            return mask
        
        try:
            mask_uint8 = (mask * 255).astype(np.uint8)
            if method == 'morphology':
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                smoothed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
                smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, kernel)
                return (smoothed > 128).astype(np.float32)
            elif method == 'gaussian':
                smoothed = cv2.GaussianBlur(mask_uint8, (kernel_size, kernel_size), 0)
                return (smoothed > 128).astype(np.float32)
            else:
                return mask
        except Exception as e:
            logger.warning(f"Error smoothing mask: {e}")
            return mask
    
    def _smooth_contour(self, contour, method: str = 'approx', **kwargs):
        """Wygładź kontur"""
        if not CV2_AVAILABLE or method == 'none':
            return contour
        
        try:
            if method == 'approx':
                epsilon = kwargs.get('epsilon_factor', 0.0002) * cv2.arcLength(contour, True)
                return cv2.approxPolyDP(contour, epsilon, True)
            elif method == 'gaussian':
                window_size = kwargs.get('window_size', 3)
                if len(contour) < window_size:
                    return contour
                contour_2d = contour.reshape(-1, 2).astype(np.float32)
                n_points = len(contour_2d)
                smoothed_points = np.zeros_like(contour_2d)
                
                for i in range(n_points):
                    window_points = []
                    for j in range(-window_size//2, window_size//2 + 1):
                        idx = (i + j) % n_points
                        window_points.append(contour_2d[idx])
                    smoothed_points[i] = np.mean(window_points, axis=0)
                
                return smoothed_points.astype(np.int32).reshape(-1, 1, 2)
            return contour
        except Exception as e:
            logger.warning(f"Error smoothing contour: {e}")
            return contour
    
    def _mask_to_polygons(self, mask: np.ndarray, original_width: int, original_height: int, min_area: int = 100) -> List[List[List[float]]]:
        """Konwertuj maskę binarną na listę polygonów"""
        try:
            polygons = []
            
            if self.smooth_mask_method != 'none':
                mask = self._smooth_mask(mask, method=self.smooth_mask_method, kernel_size=3)
            
            if CV2_AVAILABLE:
                mask_uint8 = (mask * 255).astype(np.uint8)
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area < min_area:
                        continue
                    
                    if self.polygon_detail_level == 'ultra':
                        epsilon_factor = 0.00005
                        max_points = 200
                    elif self.polygon_detail_level == 'high':
                        epsilon_factor = 0.0002
                        max_points = 100
                    elif self.polygon_detail_level == 'medium':
                        epsilon_factor = 0.0005
                        max_points = 50
                    else:
                        epsilon_factor = 0.001
                        max_points = 30
                    
                    if self.smooth_contour_method != 'none':
                        if self.smooth_contour_method == 'approx':
                            contour = self._smooth_contour(contour, method='approx', epsilon_factor=epsilon_factor)
                        elif self.smooth_contour_method == 'gaussian':
                            contour = self._smooth_contour(contour, method='gaussian', window_size=3)
                    
                    if len(contour) > max_points:
                        step = len(contour) // max_points
                        contour = contour[::step]
                    
                    polygon_points = []
                    for point in contour:
                        try:
                            if len(point.shape) == 3 and point.shape[0] == 1:
                                x, y = point[0]
                            elif len(point.shape) == 2 and point.shape[0] == 1:
                                x, y = point[0]
                            elif len(point.shape) == 1 and len(point) == 2:
                                x, y = point
                            else:
                                continue
                            
                            x_pct = float((x / original_width) * 100)
                            y_pct = float((y / original_height) * 100)
                            polygon_points.append([x_pct, y_pct])
                        except Exception:
                            continue
                    
                    if len(polygon_points) >= 3:
                        polygons.append(polygon_points)
            else:
                y_coords, x_coords = np.where(mask > 0)
                if len(x_coords) > 0 and len(y_coords) > 0:
                    x_min, x_max = float(x_coords.min()), float(x_coords.max())
                    y_min, y_max = float(y_coords.min()), float(y_coords.max())
                    
                    polygon_points = [
                        [float((x_min / original_width) * 100), float((y_min / original_height) * 100)],
                        [float((x_max / original_width) * 100), float((y_min / original_height) * 100)],
                        [float((x_max / original_width) * 100), float((y_max / original_height) * 100)],
                        [float((x_min / original_width) * 100), float((y_max / original_height) * 100)]
                    ]
                    polygons.append(polygon_points)
            
            return polygons
        except Exception as e:
            logger.error(f"Error converting mask to polygons: {e}")
            return []
    
    def _mask_to_rle(self, mask: np.ndarray) -> List[int]:
        """
        Konwertuj maskę binarną do RLE dla Label Studio.
        """
        from label_studio_converter import brush
        
        mask_255 = (mask > 0.5).astype(np.uint8) * 255
        rle = brush.mask2rle(mask_255)
        logger.debug(f"RLE generated using label_studio_converter.brush, length: {len(rle)}")
        return rle
    
    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs):
        """Główna funkcja predykcji dla Label Studio"""
        logger.info(f"Received {len(tasks)} tasks for Segformer prediction")
        
        if self.model is None or self.processor is None:
            logger.error("Segformer model or processor not loaded")
            return [{'result': [], 'score': 0.0, 'model_version': self.get_model_version_str()} for _ in tasks]
        
        predictions = []
        
        for task in tasks:
            try:
                image = self._get_image_from_task(task)
                # Upewnij się, że obraz jest w trybie RGB
                if image.mode != 'RGB':
                    logger.info(f"Converting image from {image.mode} to RGB")
                    image = image.convert('RGB')
                logger.info(f"Processing image: size={image.size}, mode={image.mode}")
                
                # Predykcja używając Segformer
                logger.info(f"Running Segformer inference with: use_clahe={self.use_clahe}, use_denoise={self.use_denoise}, "
                           f"background_threshold={self.background_threshold}, vessel_threshold={self.vessel_threshold}")
                
                pred_mask, probs = predict_pil_image_with_thresholding(
                    model=self.model,
                    image=image,
                    processor=self.processor,
                    device=self.device,
                    return_probs=True,
                    use_threshold=True,
                    background_threshold=self.background_threshold,
                    vessel_threshold=self.vessel_threshold,
                    use_denoise=self.use_denoise,
                    denoise_method=self.denoise_method,
                    use_clahe=self.use_clahe,
                    clahe_space=self.clahe_space,
                    clahe_clip=self.clahe_clip,
                    clahe_grid=self.clahe_grid
                )
                
                # Konwertuj tensor na numpy - zachowaj pełną maskę multiclass
                # WAŻNE: pred_mask jest JUŻ przeskalowana do oryginalnego rozmiaru w predict_pil_image_with_thresholding
                if isinstance(pred_mask, torch.Tensor):
                    pred_mask_np = pred_mask.numpy().astype(np.uint8)
                else:
                    pred_mask_np = pred_mask.astype(np.uint8)
                
                logger.info(f"Prediction mask shape: {pred_mask_np.shape}, unique values: {np.unique(pred_mask_np).tolist()}")
                
                # DEBUG: Zapisz maskę do pliku PNG dla weryfikacji
                try:
                    debug_mask_path = '/app/image/debug_mask.png'
                    # Skaluj wartości maski do 0-255 dla wizualizacji (unikaj overflow)
                    mask_vis = ((pred_mask_np.astype(np.int32) * 10) % 256).astype(np.uint8)
                    Image.fromarray(mask_vis).save(debug_mask_path)
                    logger.info(f"DEBUG: Saved prediction mask to {debug_mask_path}")
                except Exception as e:
                    logger.warning(f"DEBUG: Could not save mask: {e}")
                
                # Maska jest już w oryginalnym rozmiarze - NIE skaluj ponownie!
                # PIL Image.size zwraca (width, height)
                original_width, original_height = image.size
                logger.info(f"Image dimensions: width={original_width}, height={original_height}")
                logger.info(f"Mask shape from predict: {pred_mask_np.shape} (should be {original_height}x{original_width})")
                
                # Walidacja wymiarów - upewnij się, że maska ma prawidłowy rozmiar
                if pred_mask_np.shape != (original_height, original_width):
                    logger.error(f"DIMENSION MISMATCH! Mask shape {pred_mask_np.shape} != expected ({original_height}, {original_width})")
                    logger.error("This may cause duplicate/shifted segments! Resizing mask to correct dimensions...")
                    
                    # W przypadku błędu wymiarów, przeskaluj raz jeszcze
                    if CV2_AVAILABLE:
                        mask_multiclass_resized = cv2.resize(
                            pred_mask_np,
                            (original_width, original_height),
                            interpolation=cv2.INTER_NEAREST
                        ).astype(np.uint8)
                    else:
                        mask_pil = Image.fromarray(pred_mask_np, mode='L')
                        mask_multiclass_resized = np.array(mask_pil.resize((original_width, original_height), Image.NEAREST)).astype(np.uint8)
                    logger.info(f"Mask resized to: {mask_multiclass_resized.shape}")
                else:
                    # Wymiary się zgadzają - użyj maski bez skalowania
                    mask_multiclass_resized = pred_mask_np
                    logger.info("Mask dimensions correct, using mask without rescaling")
                
                # Oblicz confidence z probs
                if isinstance(probs, torch.Tensor):
                    probs_np = probs.numpy()
                else:
                    probs_np = probs
                
                # Dla multiclass: utwórz RLE dla każdej klasy osobno
                results = []
                
                # Loguj unikalne klasy w masce
                unique_classes = np.unique(mask_multiclass_resized)
                logger.info(f"Unique classes in mask: {unique_classes.tolist()}")
                logger.info(f"Mask shape: {mask_multiclass_resized.shape}, dtype: {mask_multiclass_resized.dtype}")
                
                # Iteruj przez wszystkie klasy (pomijając tło = klasa 0)
                for class_id in range(1, self.num_classes):
                    # Utwórz binarną maskę dla tej klasy
                    class_mask = (mask_multiclass_resized == class_id).astype(np.float32)
                    
                    # Sprawdź czy są jakieś piksele tej klasy PRZED czyszczeniem
                    if class_mask.sum() == 0:
                        logger.debug(f"Class {class_id}: No pixels found, skipping")
                        continue
                    
                    logger.debug(f"Class {class_id}: Found {int(class_mask.sum())} pixels before cleaning")
                    
                    # Usuń małe komponenty
                    class_mask_clean = self._remove_small_components(
                        class_mask,
                        min_size=self.min_component_size
                    )
                    
                    # Sprawdź czy są jakieś piksele tej klasy PO czyszczeniu
                    if class_mask_clean.sum() == 0:
                        logger.debug(f"Class {class_id}: No pixels after component removal, skipping")
                        continue
                    
                    logger.debug(f"Class {class_id}: {int(class_mask_clean.sum())} pixels after cleaning")
                    
                    # Konwertuj na RLE (lista liczb, column-major order)
                    rle = self._mask_to_rle(class_mask_clean)
                    
                    # Loguj RLE dla debugowania
                    logger.debug(f"RLE for class {class_id}: counts length: {len(rle)}, first 10: {rle[:10]}")
                    
                    # Pobierz confidence dla tej klasy
                    if class_id < probs_np.shape[0]:
                        class_confidence = float(probs_np[class_id, :, :].max())
                    else:
                        class_confidence = 0.5
                    
                    # Mapowanie class_id na etykiety Label Studio (format "N: nazwa")
                    CLASS_ID_TO_LABEL = {
                        1: "1: RCA prox",
                        2: "2: RCA mid",
                        3: "3: RCA dist",
                        4: "4: PDA",
                        5: "5: LM",
                        6: "6: LAD prox",
                        7: "7: LAD mid",
                        8: "8: LAD apical",
                        9: "9: D1",
                        10: "10: D1 branch (9a)",
                        11: "11: D2",
                        12: "12: D2 branch (10a)",
                        13: "13: LCx prox (11)",
                        14: "14: OM1 (12)",
                        15: "15: OM1 branch (12a)",
                        16: "16: LCx mid (13)",
                        17: "17: OM2 (14)",
                        18: "18: OM2 branch (14a)",
                        19: "19: LCx dist (15)",
                        20: "20: PLB (16)",
                        21: "21: PLB branch (16a)",
                        22: "22: PLB branch (16b)",
                        23: "23: PLB branch (16c)",
                        24: "24: OM1 branch (12b)",
                        25: "25: OM2 branch (14b)",
                        26: "26: stenosis",
                    }
                    
                    # Użyj mapowania lub fallback na "class_id: unknown"
                    prediction_class = CLASS_ID_TO_LABEL.get(class_id, f"{class_id}: unknown")
                    
                    logger.info(f"Segformer: Class {class_id} -> '{prediction_class}' (RLE length: {len(rle)}, mask pixels: {int(class_mask_clean.sum())})")
                    
                    # Utwórz wynik w formacie RLE dla Label Studio (zawsze brushlabels)
                    # WAŻNE: Label Studio wymaga:
                    # - id: unikalny identyfikator regionu
                    # - original_width, original_height: wymiary obrazu
                    # - image_rotation: rotacja obrazu (zwykle 0)
                    # - value.format: 'rle'
                    # - value.rle: tablica [start, length, start, length, ...]
                    # - value.brushlabels: lista etykiet
                    result_type = 'brushlabels'
                    result_value = {
                        'format': 'rle',
                        'rle': rle,
                        'brushlabels': [prediction_class]
                    }
                    
                    result = {
                        'id': str(uuid.uuid4())[:8],  # Unikalny ID dla regionu
                        'from_name': self.from_name,
                        'to_name': self.to_name,
                        'type': result_type,
                        'score': class_confidence,
                        'original_width': original_width,
                        'original_height': original_height,
                        'image_rotation': 0,
                        'value': result_value
                    }
                    results.append(result)
                
                # Oblicz ogólny confidence (max z wszystkich klas)
                if self.num_classes > 2:
                    vessel_probs = probs_np[1:, :, :]
                    overall_confidence = float(vessel_probs.max())
                else:
                    overall_confidence = float(probs_np[1, :, :].max()) if probs_np.shape[0] > 1 else 0.5
                
                if len(results) > 0:
                    prediction = {
                        'result': results,
                        'score': overall_confidence,
                        'model_version': self.get_model_version_str()
                    }
                else:
                    prediction = {
                        'result': [],
                        'score': overall_confidence,
                        'model_version': self.get_model_version_str()
                    }
                
                predictions.append(prediction)
                total_positive_pixels = (mask_multiclass_resized > 0).sum()
                logger.info(f"Segformer prediction completed - confidence: {overall_confidence:.4f}, "
                           f"classes found: {len(results)}, total positive pixels: {total_positive_pixels}")
                
                # Loguj przykładowy wynik dla debugowania
                if results:
                    sample_result = results[0]
                    logger.info(f"Sample result: id={sample_result.get('id')}, from_name={sample_result.get('from_name')}, "
                               f"to_name={sample_result.get('to_name')}, type={sample_result.get('type')}, "
                               f"original_width={sample_result.get('original_width')}, original_height={sample_result.get('original_height')}")
                    logger.info(f"Sample value: format={sample_result['value'].get('format')}, "
                               f"brushlabels={sample_result['value'].get('brushlabels')}, rle_len={len(sample_result['value'].get('rle', []))}")
                
            except Exception as e:
                logger.error(f"Error in Segformer prediction: {e}")
                import traceback
                traceback.print_exc()
                
                predictions.append({
                    'result': [],
                    'score': 0.0,
                    'model_version': self.get_model_version_str()
                })
        
        return predictions
    
    def fit(self, event, data, **kwargs):
        """Placeholder dla trenowania modelu"""
        logger.info(f"Fit called with event: {event}")
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Zwróć informacje o modelu"""
        return {
            'model_type': self.model_type,
            'model_size': self.model_size,
            'model_path': self.model_path,
            'resolution': self.resolution,
            'num_classes': self.num_classes,
            'background_threshold': self.background_threshold,
            'vessel_threshold': self.vessel_threshold,
            'device': str(self.device),
            'model_loaded': self.model is not None,
            'processor_loaded': self.processor is not None,
            'use_clahe': self.use_clahe,
            'use_denoise': self.use_denoise,
            'classes': self.label_schema_classes or ['coronary_artery']
        }


# Punkt wejścia dla Label Studio ML Backend
def create_model_backend(**kwargs):
    """Factory function dla tworzenia instancji modelu"""
    return CoronarySegmentationModel(**kwargs)
