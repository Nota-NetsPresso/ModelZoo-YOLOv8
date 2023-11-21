import os
import argparse
import shutil

import yaml
from loguru import logger
from netspresso.compressor import ModelCompressor, Task, Framework

from ultralytics import YOLO, YOLO_netspresso


def parse_args():
    parser = argparse.ArgumentParser()

    """
        Common arguments
    """
    parser.add_argument('-n', '--name', type=str, default='yolov8n', help='model name')
    parser.add_argument('-w', '--weights', type=str, default='./yolov8n.pt', help='weights path')
    parser.add_argument('--config', type=str, default='', help='custom config path')

    """
        Compression arguments
    """
    parser.add_argument("--compression_method", type=str, choices=["PR_L2", "PR_GM", "PR_NN", "PR_ID", "FD_TK", "FD_CP", "FD_SVD"], default="PR_L2")
    parser.add_argument("--recommendation_method", type=str, choices=["slamp", "vbmf"], default="slamp")
    parser.add_argument("--compression_ratio", type=float, default=0.3)
    parser.add_argument("-m", "--np_email", help="NetsPresso login e-mail", type=str)
    parser.add_argument("-p", "--np_password", help="NetsPresso login password", type=str)

    """
        Fine-tuning arguments
    """
    parser.add_argument('--data', type=str, default='./ultralytics/datasets/coco128.yaml', help='model.yaml path')
    parser.add_argument('--epochs', type=int, default=100, help='retrain epoch')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    model = YOLO(args.weights)
    model_args = model.model.args

    """ 
        Convert YOLOv8 model to fx 
    """
    logger.info("yolov8 to fx graph start.")

    # save model.fx and netpresso_head_meta.json
    model.export_netspresso()

    logger.info("yolov8 to fx graph end.")
    
    """
        Model compression - recommendation compression 
    """
    logger.info("Compression step start.")
    
    compressor = ModelCompressor(email=args.np_email, password=args.np_password)

    UPLOAD_MODEL_NAME = args.name
    TASK = Task.OBJECT_DETECTION
    FRAMEWORK = Framework.PYTORCH
    UPLOAD_MODEL_PATH = 'model_fx.pt'
    INPUT_SHAPES = [{"batch": 1, "channel": 3, "dimension": (model_args['imgsz'], model_args['imgsz'])}]
    upload_model = compressor.upload_model(
        model_name=UPLOAD_MODEL_NAME,
        task=TASK,
        framework=FRAMEWORK,
        file_path=UPLOAD_MODEL_PATH,
        input_shapes=INPUT_SHAPES,
    )

    COMPRESSION_METHOD = args.compression_method
    RECOMMENDATION_METHOD = args.recommendation_method
    RECOMMENDATION_RATIO = args.compression_ratio
    COMPRESSED_MODEL_NAME = f'{UPLOAD_MODEL_NAME}_{COMPRESSION_METHOD}_{RECOMMENDATION_RATIO}'
    OUTPUT_PATH = COMPRESSED_MODEL_NAME + '.pt'
    compressed_model = compressor.recommendation_compression(
        model_id=upload_model.model_id,
        model_name=COMPRESSED_MODEL_NAME,
        compression_method=COMPRESSION_METHOD,
        recommendation_method=RECOMMENDATION_METHOD,
        recommendation_ratio=RECOMMENDATION_RATIO,
        output_path=OUTPUT_PATH,
    )

    logger.info("Compression step end.")

    """
        Retrain YOLOv8 model
    """
    logger.info("Fine-tuning step start.")

    if args.config == '':
        model.ckpt['train_args']['device'] = ''
        with open('tmp_cfg.yaml', 'w') as f:
            yaml.dump(model.ckpt['train_args'], f)
    else:
        shutil.copy(args.config, 'tmp_cfg.yaml')

    lr = model.ckpt['train_args']['lr0'] * 0.1
    model = YOLO_netspresso(OUTPUT_PATH, './netspresso_head_meta.json', model_args['task'] + '_retraining', 'tmp_cfg.yaml')
    os.remove('tmp_cfg.yaml')

    model.train(data=args.data, epochs=args.epochs, lr0=lr, device=args.device)
    metrics = model.val(data=args.data)

    logger.info("Fine-tuning step end.")

    """
        Export YOLOv8 model to onnx
    """
    logger.info("Export model to onnx format step start.")

    best_model = YOLO(model.trainer.save_dir / 'weights' / 'best.pt')
    path = best_model.export(format="onnx")

    shutil.copy(path, COMPRESSED_MODEL_NAME + '.onnx')
    logger.info(f'=> saving model to {COMPRESSED_MODEL_NAME}.onnx')

    logger.info("Export model to onnx format step end.")
