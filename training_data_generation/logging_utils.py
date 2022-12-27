import logging
from logging.handlers import QueueHandler, QueueListener
from multiprocessing import Queue


def initialise_worker(queue: Queue) -> None:
    # Initialise QueueHandler with given Queue and set logging level
    queue_handler = QueueHandler(queue)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(queue_handler)


def initialise_logger():
    queue = Queue()
    handler = logging.FileHandler("dataset_creation_log.txt")
    handler.setFormatter(
        logging.Formatter("%(levelname)s: %(asctime)s - %(process)s - %(message)s")
    )
    # queue_listener gets records from the queue and sends them to the handler
    queue_listener = QueueListener(queue, handler)
    queue_listener.start()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # add the handler to the logger so records from this process are handled
    logger.addHandler(handler)
    return queue_listener, queue
