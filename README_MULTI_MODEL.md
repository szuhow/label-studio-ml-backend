# Multi-Model Coronary Segmentation Backend

## Obsługa wielu modeli .pth

Ten backend został rozszerzony o możliwość obsługi **wielu modeli PyTorch (.pth)** jednocześnie przez różne endpointy.

## Struktura katalogów

```
label-studio-ml-backend/
├── _wsgi_multi.py          # Nowy WSGI z obsługą wielu modeli
├── models_config.json      # Konfiguracja modeli
├── models/                 # Katalog z plikami .pth
│   ├── best_attention_resunet_384.pth
│   ├── best_resunet_512.pth
│   ├── best_unet_256.pth
│   └── best_attention_resunet_640.pth
└── docker-compose.yml      # Zaktualizowana konfiguracja Docker
```

## Konfiguracja modeli

### Opcja 1: Plik JSON (zalecane)

Stwórz plik `models_config.json`:

```json
{
  "models": {
    "model_384_attention": {
      "model_path": "/app/models/best_attention_resunet_384.pth",
      "model_type": "attention_resunet",
      "resolution": 384,
      "threshold": 0.5,
      "min_component_size": 300,
      "endpoint": "/",
      "description": "Główny model"
    },
    "model_512_resunet": {
      "model_path": "/app/models/best_resunet_512.pth",
      "model_type": "resunet",
      "resolution": 512,
      "threshold": 0.45,
      "min_component_size": 250,
      "endpoint": "/model_512",
      "description": "Model wysokiej rozdzielczości"
    }
  },
  "default_model": "model_384_attention"
}
```

### Opcja 2: Zmienne środowiskowe

```bash
export MODELS_CONFIG_JSON='{"models": {...}, "default_model": "..."}'
```

### Opcja 3: Auto-discovery

Umieść pliki `.pth` w katalogu `models/` - zostaną automatycznie wykryte:

```bash
export MODELS_DIR=/app/models
```

## Uruchomienie

### Docker Compose (zalecane)

1. **Umieść modele** w katalogu `models/`:
   ```bash
   mkdir models
   cp your_model1.pth models/
   cp your_model2.pth models/
   ```

2. **Edytuj konfigurację** w `models_config.json`

3. **Uruchom**:
   ```bash
   docker-compose up --build
   ```

### Lokalne uruchomienie

```bash
# Testowanie konfiguracji
python _wsgi_multi.py --check

# Uruchomienie serwera
python _wsgi_multi.py --port 9090 --models-config models_config.json
```

## Endpointy

Po uruchomieniu dostępne będą następujące endpointy:

### Informacyjne
- `GET /models` - Lista dostępnych modeli
- `GET /health` - Status serwera

### Predykcje
- `POST /` - Predykcja głównym modelem (default)
- `POST /model_512` - Predykcja modelem 512px
- `POST /model_fast` - Predykcja szybkim modelem
- `POST /model_precise` - Predykcja precyzyjnym modelem

## Przykłady użycia

### Sprawdzenie dostępnych modeli

```bash
curl http://localhost:9090/models
```

Odpowiedź:
```json
{
  "available_models": {
    "model_384_attention": {
      "endpoint": "/",
      "model_type": "attention_resunet",
      "resolution": 384,
      "description": "Główny model",
      "status": "loaded"
    },
    "model_512_resunet": {
      "endpoint": "/model_512",
      "model_type": "resunet", 
      "resolution": 512,
      "description": "Model wysokiej rozdzielczości",
      "status": "loaded"
    }
  },
  "default_model": "model_384_attention",
  "total_models": 2
}
```

### Predykcja głównym modelem

```bash
curl -X POST http://localhost:9090/predict \
  -H "Content-Type: application/json" \
  -d '{"tasks": [{"data": {"image": "data:image/jpeg;base64,..."}}]}'
```

### Predykcja konkretnym modelem

```bash
curl -X POST http://localhost:9090/model_512/predict \
  -H "Content-Type: application/json" \
  -d '{"tasks": [{"data": {"image": "data:image/jpeg;base64,..."}}]}'
```

## Konfiguracja w Label Studio

W Label Studio dodaj backend ML z odpowiednim URL:

1. **Główny model**: `http://localhost:9090`
2. **Model 512px**: `http://localhost:9090/model_512`
3. **Szybki model**: `http://localhost:9090/model_fast`

## Zalety tego rozwiązania

✅ **Prosta implementacja** - minimalne zmiany w istniejącym kodzie  
✅ **Elastyczność** - łatwe dodawanie nowych modeli  
✅ **Separacja** - każdy model ma własny endpoint  
✅ **Kompatybilność** - działa z istniejącą klasą `CoronarySegmentationModel`  
✅ **Auto-discovery** - automatyczne wykrywanie modeli w katalogu  
✅ **Konfigurowalność** - różne parametry dla każdego modelu  

## Uwagi

- Każdy model jest ładowany do pamięci przy starcie
- Więcej modeli = większe zużycie RAM
- Pierwszy model jest domyślny (endpoint `/`)
- Modele muszą być kompatybilne z funkcjami z `predictfn.py`

## Debugowanie

```bash
# Sprawdzenie czy modele się ładują
python _wsgi_multi.py --check --models-config models_config.json

# Verbose logging
LOG_LEVEL=DEBUG python _wsgi_multi.py
```
