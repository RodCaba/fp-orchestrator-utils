import logging
import torch
from ..src.data_loader import DataLoader
from ..src.har_inference import HARModel, HARTrainer

logger = logging.getLogger(__name__)

def setup_har_model_commands(subparsers):
    """ Setup HAR model-related CLI commands. """
    har_parser = subparsers.add_parser('har_model', help='HAR Model utilities')
    har_subparsers = har_parser.add_subparsers(dest='har_model_command', required=True)
    
    # Train command
    train_parser = har_subparsers.add_parser('train', help='Train HAR model')
    train_parser.add_argument('--export-onnx', action='store_true', help='Export trained model to ONNX format')
    train_parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    train_parser.add_argument('--use-local-data', action='store_true', help='Use local data instead of S3, make sure data is in the local directory')
    train_parser.add_argument('--save-data-locally', action='store_true', help='Save loaded S3 data locally')
    train_parser.add_argument('--clean-training', action='store_true', default=False, help='Clean training, ignore previous checkpoints')
    train_parser.set_defaults(func=cmd_train_har_model)

def cmd_train_har_model(args):
    """ Train HAR model command. """
    try:
        data_loader = DataLoader()
        if args.use_local_data:
            raw_data = data_loader.load_local_data()
            logger.info(f"Loaded {len(raw_data)} data files from local directory")
        else:
            raw_data = data_loader.load_data_from_s3(save_locally=args.save_data_locally)
            logger.info(f"Loaded {len(raw_data)} data files from S3")
        if not raw_data:
            logger.error("No data loaded from S3. Aborting training.")
            return 1
        
        features, labels = data_loader.preprocess_data(raw_data)
        # Determine number of classes
        num_classes = len(set(labels))
        logger.info(f"Number of classes: {num_classes}")

        # Create model and trainer
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")

        model = HARModel(num_classes)
        trainer = HARTrainer(model, device)

        # Prepare data
        train_loader, val_loader = trainer.prepare_data(features, labels)
        # Get clean_training flag from args if provided, default to False
        clean_training = getattr(args, 'clean_training', False)
        resume_training = not clean_training  # Set resume_training based on clean_training
        trainer.train(
            train_loader,
            val_loader,
            epochs=args.epochs if hasattr(args, 'epochs') else 50,
            resume_from_checkpoint=resume_training
        )
        

        logger.info("Training process completed.")

        if args.export_onnx:
            trainer.export_to_onnx("har_model.onnx")
            logger.info("Model exported to ONNX")

        logger.info("HAR model training completed successfully")
        return 0
    except Exception as e:
        logger.error(f"Error during HAR model training: {e}")
        return 1