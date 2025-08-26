import logging
from ..src.data_loader import DataLoader

logger = logging.getLogger(__name__)

def setup_har_model_commands(subparsers):
    """ Setup HAR model-related CLI commands. """
    har_parser = subparsers.add_parser('har_model', help='HAR Model utilities')
    har_subparsers = har_parser.add_subparsers(dest='har_model_command', required=True)
    
    # Train command
    train_parser = har_subparsers.add_parser('train', help='Train HAR model')
    train_parser.set_defaults(func=cmd_train_har_model)
    
    """ # Evaluate command
    evaluate_parser = har_subparsers.add_parser('evaluate', help='Evaluate HAR model')
    evaluate_parser.set_defaults(func=cmd_evaluate_har_model)
    
    # Predict command
    predict_parser = har_subparsers.add_parser('predict', help='Predict using HAR model')
    predict_parser.set_defaults(func=cmd_predict_har_model) """

def cmd_train_har_model(args):
    """ Train HAR model command. """
    try:
        data_loader = DataLoader()
        raw_data = data_loader.load_data_from_s3()
        logger.info(f"Loaded {len(raw_data)} data files from S3")
        if not raw_data:
            logger.error("No data loaded from S3. Aborting training.")
            return 1
        
        features, labels = data_loader.preprocess_data(raw_data)
        logger.info(f"Preprocessed features and labels: {features}, {labels}")
        return 0
    except Exception as e:
        logger.error(f"Error during HAR model training: {e}")
        return 1