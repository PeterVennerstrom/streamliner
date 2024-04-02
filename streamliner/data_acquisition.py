from multiprocessing import Manager, Process
from queue import Empty


class DataPrefetch:
    """
    A data prefetching class that supports two modes:

    1. Iterable Mode: When initialized without a manager, it acts as a standard
       iterable that prefetches data using multiple processes. This mode is suitable
       for simple, sequential data processing tasks to speed up development.

       Example usage:
       --------------
       data_paths = glob.glob("./my_images/*.jpg")
       prefetch = DataPrefetch(data_paths, cv2.imread)
       for loaded_image, image_src in prefetch:
           pass

    2. Queue Filling Mode: When initialized with a multiprocessing.Manager, it fills
       a queue with prefetched data without supporting iteration. This mode is
       designed for integration with other components that consume data asynchronously,
       such as a AccelProcessMap class for inference.

       DataPrefetch adapts its behavior based on the presence of a Manager
       at initialization.

    Parameters:
    -----------
    data_sources (list): A list of sources from which data will be loaded.
    data_loader_callable (callable): A function to load data given a source.
    manager (optional): An instance of multiprocessing.Manager for queue-based operation.
    worker_count (int, optional): The number of worker processes for data loading.
    queue_max_size (int, optional): The maximum size of the data queue.
    """

    def __init__(
        self,
        data_sources,
        data_loader_callable,
        manager=None,
        worker_count=2,
        queue_max_size=50,
    ):
        self.is_iterable_mode = (
            not manager
        )  # Automatically determine mode based on manager presence.
        manager = manager if manager else Manager()
        self.data_source_list = manager.list(data_sources)
        self.total_items = len(data_sources)
        self.data_loader_callable = data_loader_callable
        self.worker_count = worker_count
        self.prefetched_data_queue = manager.Queue(queue_max_size)

        if not self.is_iterable_mode:
            self.start_workers()

    def start_workers(self):
        for _ in range(self.worker_count):
            worker = Process(target=self.worker_task)
            worker.daemon = True
            worker.start()

    def worker_task(self):
        while self.data_source_list:
            try:
                src = self.data_source_list.pop(0)  # Safely pop from the beginning
                loaded_data = self.data_loader_callable(src)
                self.prefetched_data_queue.put((loaded_data, src))
            except IndexError:  # No more data to acquire
                break

    def __iter__(self):
        if not self.is_iterable_mode:
            raise RuntimeError(
                "Iteration is not supported when using the queue-filling mode."
            )
        self.start_workers()
        return self

    def __next__(self):
        if not self.is_iterable_mode:
            raise StopIteration
        try:
            return self.prefetched_data_queue.get(timeout=1)  # Adjust timeout as needed
        except Empty:
            raise StopIteration

    def __len__(self):
        return self.total_items
