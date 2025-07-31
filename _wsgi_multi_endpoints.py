import os
import argparse
import json
import logging
import logging.config
from flask import Flask, request, jsonify

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
        return {
            'model_path': os.getenv('MODEL_PATH', ''),
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
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    
    models_config_env = os.getenv('MODELS_CONFIG_JSON')
    if models_config_env:
        try:
            return json.loads(models_config_env)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in MODELS_CONFIG_JSON: {e}")
    
    models_dir = os.getenv('MODELS_DIR', os.path.join(os.path.dirname(__file__), 'models'))
    
    if os.path.exists(models_dir):
        return auto_discover_models(models_dir)
    
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
        
        # Rozpoznaj typ modelu na podstawie nazwy pliku
        model_type = 'attention_resunet'  # domyślny
        name_lower = model_name.lower()
        
        if 'deep_resunet' in name_lower or 'deepresunet' in name_lower:
            model_type = 'deep_resunet'
        elif 'attention_resunet' in name_lower or 'attentionresunet' in name_lower or 'attention' in name_lower:
            model_type = 'attention_resunet'
        elif 'resunetpp' in name_lower or 'resunet++' in name_lower or 'resunetplusplus' in name_lower:
            model_type = 'resunetpp'
        elif 'resunet' in name_lower:
            model_type = 'resunet'
        elif 'enhanced_unet' in name_lower or 'enhancedunet' in name_lower:
            model_type = 'enhanced_unet'
        elif 'unet' in name_lower:
            model_type = 'unet'
        
        resolution = 384
        for res in [256, 320, 384, 512, 640]:
            if str(res) in model_name:
                resolution = res
                break
        
        endpoint = f"/model_{i+1}" if i > 0 else "/"
        
        models_config["models"][model_name] = {
            "model_path": str(model_file),
            "model_type": model_type,
            "resolution": resolution,
            "threshold": 0.5,
            "min_component_size": 300,
            "smooth_mask_method": "morphology",
            "smooth_contour_method": "approx", 
            "polygon_detail_level": "high",
            "endpoint": endpoint,
            "description": f"Auto-discovered: {model_name}"
        }
        
        if models_config["default_model"] is None:
            models_config["default_model"] = model_name
    
    logger.info(f"Discovered {len(models_config['models'])} models")
    return models_config


def create_model_wrapper_class(model_name, model_config):
    """Stwórz wrapper klasę dla konkretnego modelu"""
    
    class ModelWrapper(CoronarySegmentationModel):
        def __init__(self, **kwargs):
            # Debug: sprawdź co jest przekazywane
            logger.info(f"🔍 ModelWrapper.__init__ for {model_name}:")
            logger.info(f"   kwargs: {kwargs}")
            logger.info(f"   model_config: {model_config}")
            
            # Połącz model_config z kwargs, gdzie model_config ma priorytet
            merged_config = {**kwargs, **model_config}
            logger.info(f"   merged_config: {merged_config}")
            
            super().__init__(**merged_config)
    
    # Dodaj unikalną nazwę do klasy
    ModelWrapper.__name__ = f"{model_name}_Wrapper"
    ModelWrapper.__qualname__ = f"{model_name}_Wrapper"
    
    return ModelWrapper


def create_multi_endpoint_app(models_config, **flask_kwargs):
    """Stwórz aplikację Flask z wieloma endpointami dla różnych modeli"""
    
    app = Flask(__name__)
    
    models = models_config.get("models", {})
    default_model = models_config.get("default_model")
    
    if not models:
        raise ValueError("No models configured!")
    
    # Dictionary przechowujący aplikacje dla poszczególnych modeli
    model_apps = {}
    model_instances = {}
    
    # Twórz aplikacje dla każdego modelu
    for model_name, model_config in models.items():
        endpoint = model_config.get("endpoint", f"/{model_name}")
        logger.info(f"Initializing model '{model_name}' for endpoint '{endpoint}'")
        
        try:
            # Stwórz wrapper klasę
            ModelWrapper = create_model_wrapper_class(model_name, model_config)
            
            # Stwórz aplikację Label Studio ML dla tego modelu
            model_app = init_app(
                model_class=ModelWrapper,
                **flask_kwargs
            )
            
            model_apps[model_name] = {
                'app': model_app,
                'endpoint': endpoint,
                'config': model_config
            }
            
            # Stwórz instancję dla informacji
            model_instances[model_name] = ModelWrapper()
            
            logger.info(f"✅ Model '{model_name}' initialized for endpoint '{endpoint}'")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize model '{model_name}': {e}")
            import traceback
            traceback.print_exc()
    
    # Sprawdź czy mamy chociaż jeden załadowany model
    if not model_apps:
        raise RuntimeError("No model apps could be created!")
    
    # Dodaj główne endpointy informacyjne
    @app.route('/models', methods=['GET'])
    def list_models():
        """Endpoint zwracający listę dostępnych modeli i ich endpointów"""
        result = {
            "available_models": {},
            "default_model": default_model,
            "total_models": len(model_apps)
        }
        
        for model_name, model_data in model_apps.items():
            config = model_data['config']
            endpoint = model_data['endpoint']
            
            result["available_models"][model_name] = {
                "endpoint": endpoint,
                "predict_url": f"{endpoint}/predict" if endpoint != "/" else "/predict",
                "model_type": config.get("model_type"),
                "resolution": config.get("resolution"),
                "description": config.get("description", ""),
                "status": "loaded"
            }
        
        return jsonify(result)
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        return jsonify({
            "status": "healthy",
            "models_loaded": len(model_apps),
            "available_endpoints": [data['endpoint'] for data in model_apps.values()],
            "timestamp": os.popen('date').read().strip()
        })
    
    # Routing dla każdego modelu
    for model_name, model_data in model_apps.items():
        model_app = model_data['app']
        endpoint = model_data['endpoint']
        
        logger.info(f"🔗 Setting up routing: Model '{model_name}' -> endpoint '{endpoint}'")
        
        # Funkcja tworząca view dla konkretnego modelu
        def create_model_view(app, name, ep):
            def model_view(path='', app=app, name=name, ep=ep):  # zamrożenie argumentów
                # Import dla każdego żądania
                from flask import request as flask_request
                
                # Debug routing info
                logger.info(f"🔄 Routing request to model: {name}, endpoint: {ep}, path: {flask_request.path}")
                
                # Przygotuj ścieżkę dla przekierowania
                new_path = flask_request.path.replace(ep.rstrip('/'), '') or '/'
                
                # Skopiuj nagłówki do zwykłego dict aby uniknąć problemu z EnvironHeaders
                headers_dict = dict(flask_request.headers)
                
                # Przekieruj całe żądanie do odpowiedniej aplikacji modelu
                with app.test_request_context(
                    path=new_path,
                    method=flask_request.method,
                    headers=headers_dict,
                    data=flask_request.get_data(),
                    query_string=flask_request.query_string
                ):
                    try:
                        response = app.full_dispatch_request()
                        return response
                    except Exception as e:
                        logger.error(f"Error in model {name} at endpoint {ep}: {e}")
                        return jsonify({"error": str(e)}), 500
            model_view.__name__ = f'{name}_view'
            return model_view
        
        # Dodaj reguły routingu
        if endpoint == "/":
            # Główny endpoint
            app.add_url_rule('/', f'model_{model_name}_root', 
                           create_model_view(model_app, model_name, endpoint), 
                           methods=['GET', 'POST'])
            app.add_url_rule('/<path:path>', f'model_{model_name}_path', 
                           create_model_view(model_app, model_name, endpoint), 
                           methods=['GET', 'POST'])
        else:
            # Dodatkowe endpointy
            endpoint_clean = endpoint.rstrip('/')
            app.add_url_rule(f'{endpoint_clean}', f'model_{model_name}_root', 
                           create_model_view(model_app, model_name, endpoint), 
                           methods=['GET', 'POST'])
            app.add_url_rule(f'{endpoint_clean}/<path:path>', f'model_{model_name}_path', 
                           create_model_view(model_app, model_name, endpoint), 
                           methods=['GET', 'POST'])
    
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multi-Endpoint Coronary Segmentation ML Backend')
    parser.add_argument('-p', '--port', dest='port', type=int, default=9090, help='Server port')
    parser.add_argument('--host', dest='host', type=str, default='0.0.0.0', help='Server host')
    parser.add_argument('-d', '--debug', dest='debug', action='store_true', help='Switch debug mode')
    parser.add_argument('--log-level', dest='log_level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default=log_level, help='Logging level')
    parser.add_argument('--models-config', dest='models_config', default=_DEFAULT_MODELS_CONFIG_PATH, help='Path to models configuration JSON file')
    parser.add_argument('--check', dest='check', action='store_true', help='Validate model instances before launching server')
    parser.add_argument('--basic-auth-user', default=os.environ.get('ML_SERVER_BASIC_AUTH_USER', None), help='Basic auth user')
    parser.add_argument('--basic-auth-pass', default=os.environ.get('ML_SERVER_BASIC_AUTH_PASS', None), help='Basic auth pass')
    
    args = parser.parse_args()

    if args.log_level:
        logging.root.setLevel(args.log_level)

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
                ModelWrapper = create_model_wrapper_class(model_name, model_config)
                model = ModelWrapper()
                print(f'✅ Model "{model_name}" created successfully')
                
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

    try:
        app = create_multi_endpoint_app(
            models_config,
            basic_auth_user=args.basic_auth_user,
            basic_auth_pass=args.basic_auth_pass
        )
        
        models = models_config.get("models", {})
        
        print(f"🏥 Starting Multi-Endpoint Coronary Segmentation ML Backend")
        print(f"📋 Loaded Models:")
        
        for model_name, model_config in models.items():
            endpoint = model_config.get("endpoint", f"/{model_name}")
            model_type = model_config.get("model_type", "unknown")
            resolution = model_config.get("resolution", "unknown")
            print(f"   • {model_name}: {endpoint} ({model_type}, {resolution}px)")
        
        print(f"🌐 Server: http://{args.host}:{args.port}")
        print(f"📡 Endpoints:")
        print(f"   • GET  /models  - Lista dostępnych modeli")
        print(f"   • GET  /health  - Health check")
        
        for model_name, model_config in models.items():
            endpoint = model_config.get("endpoint", f"/{model_name}")
            predict_url = f"{endpoint}/predict" if endpoint != "/" else "/predict"
            print(f"   • POST {predict_url} - Predykcja ({model_name})")
        
        app.run(host=args.host, port=args.port, debug=args.debug)
        
    except Exception as e:
        logger.error(f"Failed to create multi-endpoint application: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

else:
    # for uWSGI/gunicorn use
    try:
        models_config = load_models_config()
        app = create_multi_endpoint_app(models_config)
    except Exception as e:
        logger.error(f"Failed to initialize multi-endpoint app for WSGI: {e}")
        kwargs = get_kwargs_from_config()
        app = init_app(model_class=CoronarySegmentationModel)
