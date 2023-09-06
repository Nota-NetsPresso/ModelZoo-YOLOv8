import argparse

from loguru import logger

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser()

    """
        Common arguments
    """
    parser.add_argument('-w', '--weights', type=str, default='./yolov8n.pt', help='weights path')


    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    model = YOLO(args.weights)

    """ 
        Convert YOLOv8 model to fx 
    """
    logger.info("yolov8 to fx graph start.")

    # save model.fx and netpresso_head_meta.json
    model.export_netspresso()

    logger.info("yolov8 to fx graph end.")
    