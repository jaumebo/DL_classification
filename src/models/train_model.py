import logging

from src.models.utils.create_model import custom_CNN_model

if __name__ == '__main__':

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    logger = logging.getLogger(__name__)
    

    model = custom_CNN_model((512,512) + (3,),4)

    logger.info(model.summary())
