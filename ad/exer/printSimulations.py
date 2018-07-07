from ds import Queue
import random


class Printer:

    def __init__(self, ppm):
        self.page_rate = ppm
        self.current_state = None
        self.remaining_time = 0

    def tick(self):
        if self.current_state:
            self.remaining_time -= 1
        if self.remaining_time <= 0:
            self.current_state = None

    def is_working(self):
        return True if self.current_state else False

    def get_new_state(self, task):
        self.current_state = task
        self.remaining_time = task.get_pages() * 60 / self.page_rate


class Task:

    def __init__(self, start_time):
        self.time_stamp = start_time
        self.pages = random.randrange(1, 21)

    def get_time(self):
        return self.time_stamp

    def get_pages(self):
        return self.pages

    def get_waiting_time(self, current_time):
        return current_time - self.time_stamp


def simulation(total_time, pages_per_minute):
    lab_printer = Printer(pages_per_minute)
    print_queue = Queue()
    waiting_time_list = []

    for current_time in range(total_time):

        if create_task():
            new_task = Task(current_time)
            print_queue.enqueue(new_task)

        if (not lab_printer.is_working()) and (print_queue.size()):
            next_task = print_queue.dequeue()
            lab_printer.get_new_state(next_task)
            waiting_time_list.append(next_task.get_waiting_time(current_time))

        lab_printer.tick()

    average_wait = sum(waiting_time_list) / len(waiting_time_list)
    print('Average Wait {:8.2f} secs {:2d} tasks remaining.'.format(average_wait, print_queue.size()))


def create_task():
    item = random.randrange(1, 181)
    return True if item == 180 else False


if __name__ == '__main__':
    for _ in range(10):
        simulation(3600, 5)
