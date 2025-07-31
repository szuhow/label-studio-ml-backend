# Multi-Model Coronary Segmentation Backend

## Obsługa wielu modeli .pth

Ten backend został rozszerzony o możliwość **konfiguracji i łatwego przełączania między różnymi modelami PyTorch (.pth)**.

## Nowe podejście - Konfigurowalny model

Zamiast uruchamiania wielu modeli jednocześnie, system pozwala na:
- **Konfigurację wielu modeli** w pliku JSON
- **Wybór aktywnego modelu** przez ustawienie `default_model`
- **Łatwe przełączanie** między modelami przez edycję konfiguracji
- **Oszczędność pamięci** - tylko jeden model w RAM na raz

## Struktura katalogów

```
label-studio-ml-backend/
├── _wsgi_multi.py          # Nowy WSGI z obsługą konfiguracji
├── models_config.json      # Konfiguracja modeli
├── models/                 # Katalog z plikami .pth
│   ├── best_attention_resunet_384.pth
│   ├── best_model.pth (attention_resunet)
│   ├── best_unet_256.pth
│   └── best_attention_resunet_640.pth
└── docker-compose.yml      # Zaktualizowana konfiguracja Docker
```

## Konfiguracja modeli

### Plik models_config.json

```json
{
  "models": {
    "model_384_attention": {
      "model_path": "/app/models/best_attention_resunet_384.pth",
      "model_type": "attention_resunet",
      "resolution": 384,
      "threshold": 0.5,
      "min_component_size": 300,
      "description": "Główny model 384px"
    },
    "model_512_resunet": {
      "model_path": "/app/models/best_model.pth",
      "model_type": "attention_resunet",
      "resolution": 512,
      "threshold": 0.45,
      "min_component_size": 250,
      "description": "Model wysokiej rozdzielczości"
    }
  },
  "default_model": "model_384_attention"
}
```

### Przełączanie między modelami

Aby zmienić aktywny model, wystarczy:

1. **Edytować `models_config.json`**:
   ```json
   "default_model": "model_512_resunet"
   ```

2. **Restart serwera**:
   ```bash
   docker-compose restart
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

### Informacyjne
- `GET /models` - Lista skonfigurowanych modeli i aktywny model
- `GET /health` - Status serwera

### Predykcje  
- `POST /predict` - Predykcja aktywnym modelem

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
      "model_type": "attention_resunet",
      "resolution": 384,
      "description": "Główny model 384px",
      "is_active": true
    },
    "model_512_resunet": {
      "model_type": "attention_resunet", 
      "resolution": 512,
      "description": "Model wysokiej rozdzielczości",
      "is_active": false
    }
  },
  "active_model": "model_384_attention",
  "total_configured": 2
}
```

### Predykcja

```bash
curl -X POST http://localhost:9090/predict \
  -H "Content-Type: application/json" \
  -d '{"tasks": [{"data": {"image": "data:image/jpeg;base64,..."}}]}'
```

## Konfiguracja w Label Studio

W Label Studio dodaj backend ML z URL: `http://localhost:9090`

## Zalety tego rozwiązania

✅ **Prostota** - jeden model aktywny na raz  
✅ **Oszczędność pamięci** - tylko aktywny model w RAM  
✅ **Łatwe przełączanie** - edycja JSON + restart  
✅ **Elastyczność** - łatwe dodawanie nowych modeli  
✅ **Kompatybilność** - działa z istniejącą klasą `CoronarySegmentationModel`  
✅ **Przewidywalność** - zawsze wiadomo, który model jest aktywny  

## Użycie z różnymi modelami

### Model szybki (256px)
```json
"default_model": "model_256_fast"
```
- Szybka inferencja
- Mniejsza dokładność
- Idealne do wstępnych adnotacji

### Model standardowy (384px)  
```json
"default_model": "model_384_attention"
```
- Kompromis szybkość/jakość
- Zalecany do większości przypadków

### Model precyzyjny (512px+)
```json
"default_model": "model_512_resunet"
```
- Wysoka dokładność
- Wolniejsza inferencja
- Idealne do finalnych adnotacji

## Debugowanie

```bash
# Sprawdzenie czy modele się ładują
python _wsgi_multi.py --check --models-config models_config.json

# Verbose logging
LOG_LEVEL=DEBUG python _wsgi_multi.py
```
