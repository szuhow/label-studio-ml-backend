import os
import argparse
import json
import logging
import logging.config

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


_DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Coronary Artery Segmentation ML Backend')
    parser.add_argument(
        '-p', '--port', dest='port', type=int, default=9090,
        help='Server port')
    parser.add_argument(
        '--host', dest='host', type=str, default='0.0.0.0',
        help='Server host')
    parser.add_argument(
        '--kwargs', '--with', dest='kwargs', metavar='KEY=VAL', nargs='+', type=lambda kv: kv.split('='),
        help='Additional CoronarySegmentationModel initialization kwargs')
    parser.add_argument(
        '-d', '--debug', dest='debug', action='store_true',
        help='Switch debug mode')
    parser.add_argument(
        '--log-level', dest='log_level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default=log_level,
        help='Logging level')
    parser.add_argument(
        '--model-dir', dest='model_dir', default=os.path.dirname(__file__),
        help='Directory where models are stored (relative to the project directory)')
    parser.add_argument(
        '--check', dest='check', action='store_true',
        help='Validate model instance before launching server')
    parser.add_argument('--basic-auth-user',
                        default=os.environ.get('ML_SERVER_BASIC_AUTH_USER', None),
                        help='Basic auth user')
    
    parser.add_argument('--basic-auth-pass',
                        default=os.environ.get('ML_SERVER_BASIC_AUTH_PASS', None),
                        help='Basic auth pass')
    
    # Dodatkowe argumenty specyficzne dla CoronarySegmentationModel
    parser.add_argument('--model-path', dest='model_path',
                        default=os.getenv('MODEL_PATH'),
                        help='Path to the trained model (.pth file)')
    parser.add_argument('--model-type', dest='model_type',
                        default=os.getenv('MODEL_TYPE', 'attention_resunet'),
                        choices=['unet', 'enhanced_unet', 'resunet', 'resunetpp', 'attention_resunet', 'deep_resunet'],
                        help='Model architecture type')
    parser.add_argument('--resolution', dest='resolution', type=int,
                        default=int(os.getenv('RESOLUTION', '384')),
                        help='Input resolution for the model')
    parser.add_argument('--threshold', dest='threshold', type=float,
                        default=float(os.getenv('THRESHOLD', '0.5')),
                        help='Binary threshold for segmentation')
    
    args = parser.parse_args()

    # setup logging level
    if args.log_level:
        logging.root.setLevel(args.log_level)

    def isfloat(value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    def parse_kwargs():
        param = dict()
        if args.kwargs:
            for k, v in args.kwargs:
                if v.isdigit():
                    param[k] = int(v)
                elif v == 'True' or v == 'true':
                    param[k] = True
                elif v == 'False' or v == 'false':
                    param[k] = False
                elif isfloat(v):
                    param[k] = float(v)
                else:
                    param[k] = v
        return param

    # Pobierz konfigurację podstawową
    kwargs = get_kwargs_from_config()

    # Nadpisz argumentami z linii poleceń
    if args.model_path:
        kwargs['model_path'] = args.model_path
    if args.model_type:
        kwargs['model_type'] = args.model_type
    if args.resolution:
        kwargs['resolution'] = args.resolution
    if args.threshold:
        kwargs['threshold'] = args.threshold

    # Dodaj dodatkowe kwargs
    kwargs.update(parse_kwargs())

    if args.check:
        print('Check "' + CoronarySegmentationModel.__name__ + '" instance creation..')
        try:
            model = CoronarySegmentationModel(**kwargs)
            print('✅ Model instance created successfully')
            
            # Test model info
            info = model.get_model_info()
            print(f'Model info: {json.dumps(info, indent=2)}')
            
        except Exception as e:
            print(f'❌ Model creation failed: {e}')
            import traceback
            traceback.print_exc()
            exit(1)

    # Utwórz aplikację
    # Przekaż konfigurację przez zmienne środowiskowe dla Label Studio ML Backend
    os.environ['MODEL_PATH'] = kwargs.get('model_path', '/home/ives/rafal/label-ml-backend/label-studio-ml-backend/ml-backend/best_attention_resunet_dice_64_25_checkpoint_epoch_25.pth')
    os.environ['MODEL_TYPE'] = kwargs.get('model_type', 'attention_resunet')
    os.environ['RESOLUTION'] = str(kwargs.get('resolution', 384))
    os.environ['THRESHOLD'] = str(kwargs.get('threshold', 0.5))
    os.environ['MIN_COMPONENT_SIZE'] = str(kwargs.get('min_component_size', 300))
    
    app = init_app(
        model_class=CoronarySegmentationModel, 
        basic_auth_user=args.basic_auth_user, 
        basic_auth_pass=args.basic_auth_pass
    )

    print(f"🏥 Starting Coronary Segmentation ML Backend")
    print(f"📋 Configuration:")
    print(f"   Model Path: {kwargs.get('model_path', 'Not set')}")
    print(f"   Model Type: {kwargs.get('model_type', 'attention_resunet')}")
    print(f"   Resolution: {kwargs.get('resolution', 384)}")
    print(f"   Threshold: {kwargs.get('threshold', 0.5)}")
    print(f"🌐 Server: http://{args.host}:{args.port}")

    app.run(host=args.host, port=args.port, debug=args.debug)

else:
    # for uWSGI/gunicorn use
    kwargs = get_kwargs_from_config()
    
    # Przekaż konfigurację przez zmienne środowiskowe
    os.environ.setdefault('MODEL_PATH', kwargs.get('model_path', ''))
    os.environ.setdefault('MODEL_TYPE', kwargs.get('model_type', 'attention_resunet'))
    os.environ.setdefault('RESOLUTION', str(kwargs.get('resolution', 384)))
    os.environ.setdefault('THRESHOLD', str(kwargs.get('threshold', 0.5)))
    os.environ.setdefault('MIN_COMPONENT_SIZE', str(kwargs.get('min_component_size', 300)))
    
    app = init_app(model_class=CoronarySegmentationModel)
