import os
import argparse
import json
import logging
import logging.config
from flask import Flask, request, jsonify
from werkzeug.routing import Rule

# Set a default log level if LOG_LEVEL is not defined
log_level = os.getenv("LOG_LEVEL", "INFO")

logging.config.dictConfig({
  "version": 1,
  "disable_existing_loggers": False,
  "formatters": {
    "standard": {
      "format": "[%(asctime)s] [%(levelname)s] [%(name)s::%(funcName)s::%(lineno)d] %(message)s"
    }
  },
  "handlers": {
    "console": {
      "class": "logging.StreamHandler",
      "level": log_level,
      "stream": "ext://sys.stdout",
      "formatter": "standard"
    }
  },
  "root": {
    "level": log_level,
    "handlers": ["console"]
  }
})

from label_studio_ml.api import init_app
from model import CoronarySegmentationModel

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')
_DEFAULT_MODELS_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'models_config.json')


def get_kwargs_from_config(config_path=_DEFAULT_CONFIG_PATH):
    """Pobierz konfigurację z pliku JSON"""
    if not os.path.exists(config_path):
        # Zwróć domyślną konfigurację dla CoronarySegmentationModel
        return {
            'model_path': os.getenv('MODEL_PATH', os.path.join(os.path.dirname(__file__), '/home/ives/rafal/label-ml-backend/label-studio-ml-backend/ml-backend/best_attention_resunet_dice_64_25_checkpoint_epoch_25.pth')),
            'model_type': os.getenv('MODEL_TYPE', 'attention_resunet'),
            'resolution': int(os.getenv('RESOLUTION', '384')),
            'threshold': float(os.getenv('THRESHOLD', '0.5')),
            'min_component_size': int(os.getenv('MIN_COMPONENT_SIZE', '300'))
        }
    
    with open(config_path) as f:
        config = json.load(f)
    assert isinstance(config, dict)
    return config


def load_models_config(config_path=_DEFAULT_MODELS_CONFIG_PATH):
    """Załaduj konfigurację wielu modeli"""
    
    # Sprawdź czy plik konfiguracji istnieje
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    
    # Sprawdź zmienną środowiskową
    models_config_env = os.getenv('MODELS_CONFIG_JSON')
    if models_config_env:
        try:
            return json.loads(models_config_env)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in MODELS_CONFIG_JSON: {e}")
    
    # Domyślna konfiguracja z auto-discovery
    models_dir = os.getenv('MODELS_DIR', os.path.join(os.path.dirname(__file__), 'models'))
    
    if os.path.exists(models_dir):
        return auto_discover_models(models_dir)
    
    # Fallback - jeden model z zmiennych środowiskowych
    model_path = os.getenv('MODEL_PATH')
    if model_path:
        return {
            "models": {
                "default": {
                    "model_path": model_path,
                    "model_type": os.getenv('MODEL_TYPE', 'attention_resunet'),
                    "resolution": int(os.getenv('RESOLUTION', '384')),
                    "threshold": float(os.getenv('THRESHOLD', '0.5')),
                    "min_component_size": int(os.getenv('MIN_COMPONENT_SIZE', '300')),
                    "endpoint": "/",
                    "description": "Default model from environment"
                }
            },
            "default_model": "default"
        }
    
    return {"models": {}, "default_model": None}


def auto_discover_models(models_dir):
    """Automatyczne wykrywanie modeli .pth w katalogu"""
    from pathlib import Path
    
    logger.info(f"Auto-discovering models in {models_dir}")
    
    model_files = list(Path(models_dir).glob("*.pth"))
    models_config = {"models": {}, "default_model": None}
    
    for i, model_file in enumerate(model_files):
        model_name = model_file.stem
        
        # Próbuj wydobyć typ modelu z nazwy pliku
        model_type = 'attention_resunet'  # domyślny
        if 'unet' in model_name.lower():
            if 'attention' in model_name.lower():
                model_type = 'attention_resunet'
            elif 'resunet' in model_name.lower():
                model_type = 'resunet'
            else:
                model_type = 'unet'
        
        # Próbuj wydobyć rozdzielczość z nazwy
        resolution = 384  # domyślna
        for res in [256, 320, 384, 512, 640]:
            if str(res) in model_name:
                resolution = res
                break
        
        endpoint = f"/model_{i+1}" if i > 0 else "/"  # Pierwszy model na głównym endpoincie
        
        models_config["models"][model_name] = {
            "model_path": str(model_file),
            "model_type": model_type,
            "resolution": resolution,
            "threshold": 0.5,
            "min_component_size": 300,
            "endpoint": endpoint,
            "description": f"Auto-discovered: {model_name}"
        }
        
        # Pierwszy model jako domyślny
        if models_config["default_model"] is None:
            models_config["default_model"] = model_name
    
    logger.info(f"Discovered {len(models_config['models'])} models")
    return models_config


def create_multi_model_app(models_config, **flask_kwargs):
    """Stwórz aplikację Flask z wieloma modelami"""
    
    # Główna aplikacja Flask
    main_app = Flask(__name__)
    
    # Słownik przechowujący aplikacje dla poszczególnych modeli
    model_apps = {}
    model_instances = {}
    
    models = models_config.get("models", {})
    default_model = models_config.get("default_model")
    
    if not models:
        raise ValueError("No models configured!")
    
    # Twórz aplikacje dla każdego modelu
    for model_name, model_config in models.items():
        logger.info(f"Initializing model '{model_name}' for endpoint '{model_config.get('endpoint', '/')}'")
        
        try:
            # Stwórz instancję modelu z odpowiednią konfiguracją
            model_instance = CoronarySegmentationModel(**model_config)
            model_instances[model_name] = model_instance
            
            # Stwórz aplikację Label Studio ML dla tego modelu
            model_app = init_app(
                model_class=lambda **kwargs: model_instance,  # Użyj gotowej instancji
                **flask_kwargs
            )
            
            model_apps[model_name] = model_app
            
            logger.info(f"✅ Model '{model_name}' initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize model '{model_name}': {e}")
            import traceback
            traceback.print_exc()
    
    # Dodaj routing do głównej aplikacji
    @main_app.route('/models', methods=['GET'])
    def list_models():
        """Endpoint zwracający listę dostępnych modeli"""
        result = {
            "available_models": {},
            "default_model": default_model,
            "total_models": len(model_instances)
        }
        
        for model_name, model_config in models.items():
            if model_name in model_instances:
                result["available_models"][model_name] = {
                    "endpoint": model_config.get("endpoint", "/"),
                    "model_type": model_config.get("model_type"),
                    "resolution": model_config.get("resolution"),
                    "description": model_config.get("description", ""),
                    "status": "loaded"
                }
        
        return jsonify(result)
    
    @main_app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        return jsonify({
            "status": "healthy",
            "models_loaded": len(model_instances),
            "timestamp": os.popen('date').read().strip()
        })
    
    # Dynamiczne routowanie do aplikacji modeli
    for model_name, model_config in models.items():
        if model_name not in model_apps:
            continue
            
        endpoint = model_config.get("endpoint", "/")
        model_app = model_apps[model_name]
        
        # Funkcja tworząca handler dla danego modelu
        def create_model_handler(app, name):
            def handle_request(path=""):
                # Przekieruj żądanie do odpowiedniej aplikacji modelu
                with app.test_request_context():
                    return app.full_dispatch_request()
            return handle_request
        
        # Dodaj reguły routingu
        if endpoint == "/":
            # Główny endpoint
            main_app.add_url_rule('/', f'model_{model_name}_root', 
                                 create_model_handler(model_app, model_name), 
                                 methods=['GET', 'POST'])
            main_app.add_url_rule('/<path:path>', f'model_{model_name}_path', 
                                 create_model_handler(model_app, model_name), 
                                 methods=['GET', 'POST'])
        else:
            # Dodatkowe endpointy
            main_app.add_url_rule(f'{endpoint}', f'model_{model_name}_root', 
                                 create_model_handler(model_app, model_name), 
                                 methods=['GET', 'POST'])
            main_app.add_url_rule(f'{endpoint}/<path:path>', f'model_{model_name}_path', 
                                 create_model_handler(model_app, model_name), 
                                 methods=['GET', 'POST'])
    
    return main_app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multi-Model Coronary Artery Segmentation ML Backend')
    parser.add_argument(
        '-p', '--port', dest='port', type=int, default=9090,
        help='Server port')
    parser.add_argument(
        '--host', dest='host', type=str, default='0.0.0.0',
        help='Server host')
    parser.add_argument(
        '--kwargs', '--with', dest='kwargs', metavar='KEY=VAL', nargs='+', type=lambda kv: kv.split('='),
        help='Additional model initialization kwargs')
    parser.add_argument(
        '-d', '--debug', dest='debug', action='store_true',
        help='Switch debug mode')
    parser.add_argument(
        '--log-level', dest='log_level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default=log_level,
        help='Logging level')
    parser.add_argument(
        '--models-config', dest='models_config', default=_DEFAULT_MODELS_CONFIG_PATH,
        help='Path to models configuration JSON file')
    parser.add_argument(
        '--models-dir', dest='models_dir', default=os.getenv('MODELS_DIR'),
        help='Directory containing model .pth files (for auto-discovery)')
    parser.add_argument(
        '--check', dest='check', action='store_true',
        help='Validate model instances before launching server')
    parser.add_argument('--basic-auth-user',
                        default=os.environ.get('ML_SERVER_BASIC_AUTH_USER', None),
                        help='Basic auth user')
    parser.add_argument('--basic-auth-pass',
                        default=os.environ.get('ML_SERVER_BASIC_AUTH_PASS', None),
                        help='Basic auth pass')
    
    args = parser.parse_args()

    # setup logging level
    if args.log_level:
        logging.root.setLevel(args.log_level)

    # Ustaw ścieżkę do katalogu modeli jeśli podana
    if args.models_dir:
        os.environ['MODELS_DIR'] = args.models_dir
    
    # Załaduj konfigurację modeli
    try:
        models_config = load_models_config(args.models_config)
        logger.info(f"Loaded configuration for {len(models_config.get('models', {}))} models")
    except Exception as e:
        logger.error(f"Failed to load models configuration: {e}")
        exit(1)

    if args.check:
        print('🔍 Checking model instances creation...')
        
        models = models_config.get("models", {})
        success_count = 0
        
        for model_name, model_config in models.items():
            print(f'Checking "{model_name}"...')
            try:
                model = CoronarySegmentationModel(**model_config)
                print(f'✅ Model "{model_name}" created successfully')
                
                # Test model info
                info = model.get_model_info()
                print(f'   Model info: {json.dumps(info, indent=2)}')
                success_count += 1
                
            except Exception as e:
                print(f'❌ Model "{model_name}" creation failed: {e}')
                import traceback
                traceback.print_exc()
        
        print(f'\n📊 Summary: {success_count}/{len(models)} models loaded successfully')
        
        if success_count == 0:
            print('❌ No models could be loaded!')
            exit(1)
        else:
            print('✅ Check completed')
            exit(0)

    # Utwórz aplikację z wieloma modelami
    try:
        app = create_multi_model_app(
            models_config,
            basic_auth_user=args.basic_auth_user,
            basic_auth_pass=args.basic_auth_pass
        )
        
        print(f"🏥 Starting Multi-Model Coronary Segmentation ML Backend")
        print(f"📋 Loaded Models:")
        
        for model_name, model_config in models_config.get("models", {}).items():
            endpoint = model_config.get("endpoint", "/")
            model_type = model_config.get("model_type", "unknown")
            resolution = model_config.get("resolution", "unknown")
            print(f"   • {model_name}: {endpoint} ({model_type}, {resolution}px)")
        
        print(f"🌐 Server: http://{args.host}:{args.port}")
        print(f"📡 Endpoints:")
        print(f"   • GET  /models  - Lista dostępnych modeli")
        print(f"   • GET  /health  - Health check")
        print(f"   • POST /        - Predykcja (domyślny model)")
        
        # Wyświetl dodatkowe endpointy
        for model_name, model_config in models_config.get("models", {}).items():
            endpoint = model_config.get("endpoint", "/")
            if endpoint != "/":
                print(f"   • POST {endpoint}     - Predykcja ({model_name})")
        
        app.run(host=args.host, port=args.port, debug=args.debug)
        
    except Exception as e:
        logger.error(f"Failed to create multi-model application: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

else:
    # for uWSGI/gunicorn use
    try:
        models_config = load_models_config()
        app = create_multi_model_app(models_config)
    except Exception as e:
        logger.error(f"Failed to initialize multi-model app for WSGI: {e}")
        # Fallback na pojedynczy model
        kwargs = get_kwargs_from_config()
        app = init_app(model_class=CoronarySegmentationModel)
