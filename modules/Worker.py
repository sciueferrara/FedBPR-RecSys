import multiprocessing


class Worker(multiprocessing.Process):
    def __init__(self, task_queue, result_queue, work, clients):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.work = work
        self.clients = clients

    def run(self):
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                # Poison pill means shutdown
                self.task_queue.task_done()
                break
            answer = self.work(self.clients, next_task)
            self.task_queue.task_done()
            self.result_queue.put(answer)
        return
