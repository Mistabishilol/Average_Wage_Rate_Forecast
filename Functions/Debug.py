import logging
import datetime

def err_log(err):
    now = datetime.datetime.now().strftime("%d.%m.%Y %H-%M-%S")

    logging.basicConfig(filename='./Errors/Errors.txt',
                        level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(name)s %(message)s'
                        )

    logger=logging.getLogger(__name__)

    return logger.error(err)