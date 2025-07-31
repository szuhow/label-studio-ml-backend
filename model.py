import os
import logging
import numpy as np
import torch

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

# Import funkcji z predict.py
import sys
from predictfn import (
    create_model, 
    preprocess_single_image, 
    load_trained_model_for_inference,
    inference_single_image,
    remove_small_components_from_mask
)

logger = logging.getLogger(__name__)


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
            'smooth_mask_method', 'smooth_contour_method', 'polygon_detail_level'
        }
        
        # Podziel kwargs na te dla klasy bazowej i te dla naszego modelu
        base_kwargs = {k: v for k, v in kwargs.items() if k not in model_specific_params}
        model_kwargs = {k: v for k, v in kwargs.items() if k in model_specific_params}
        
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
        
        # Debug info
        logger.info(f"Model configuration:")
        logger.info(f"  model_path: {self.model_path}")
        logger.info(f"  model_type: {self.model_type}")
        logger.info(f"  resolution: {self.resolution}")
        logger.info(f"  threshold: {self.threshold}")
        logger.info(f"  smooth_mask_method: {self.smooth_mask_method}")
        logger.info(f"  smooth_contour_method: {self.smooth_contour_method}")
        logger.info(f"  polygon_detail_level: {self.polygon_detail_level}")
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
        try:
            logger.info(f"Attempting to load model from: {self.model_path}")
            logger.info(f"File exists check: {os.path.exists(self.model_path)}")
            logger.info(f"Current working directory: {os.getcwd()}")
            
            # Sprawdź czy plik istnieje
            if os.path.exists(self.model_path):
                logger.info(f"Model file found, loading...")
                self.model, self.device = load_trained_model_for_inference(
                    self.model_path, 
                    self.model_type
                )
                logger.info(f"Model loaded successfully from {self.model_path}")
                logger.info(f"Device: {self.device}")
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
                
                # Znajdź komponenty połączone
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                    mask_uint8, connectivity=8
                )
                
                # Stwórz nową pustą maskę
                mask_cleaned = np.zeros_like(mask, dtype=np.uint8)
                
                # Zostaw tylko duże komponenty
                for i in range(1, num_labels):  # pomiń tło (i=0)
                    if stats[i, cv2.CC_STAT_AREA] >= min_size:
                        mask_cleaned[labels == i] = 1
                
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
                        except Exception as point_error:
                            logger.warning(f"Error processing point {point}: {point_error}")
                            continue
                    
                    # Dodaj polygon tylko jeśli ma przynajmniej 3 punkty
                    if len(polygon_points) >= 3:
                        polygons.append(polygon_points)
                        logger.info(f"Created polygon with {len(polygon_points)} points (detail level: {self.polygon_detail_level})")
                        
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
                    apply_preprocessing=True
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


# Punkt wejścia dla Label Studio ML Backend
def create_model_backend(**kwargs):
    """Factory function dla tworzenia instancji modelu"""
    return CoronarySegmentationModel(**kwargs)
