import random
from ds import Queue


class Printer:

    def __init__(self, ppm):
        self.pagerate = ppm
        self.currentTask = None
        self.timeRemaining = None

    def tick(self):
        if self.currentTask:
            self.timeRemaining -= 1
            if self.timeRemaining <= 0:
                self.currentTask = None

    def busy(self):
        if self.currentTask:
            return True
        else:
            return None

    def start_next(self, newtask):
        self.currentTask = newtask
        self.timeRemaining = newtask.getPages() * 60 / self.pagerate


class Task:

    def __init__(self, time):
        self.timestamp = time
        self.pages = random.randrange(1, 21)

    def getStamp(self):
        return self.timestamp

    def getPages(self):
        return self.pages

    def waitTime(self, currenttime):
        return currenttime - self.timestamp


def simulation(numSeconds, pagesPerMinute):
    labprinter = Printer(pagesPerMinute)
    printQueue = Queue()
    waitingtimes = []

    for currentSecond in range(numSeconds):

        if newPrintTask():
            task = Task(currentSecond)
            printQueue.enqueue(task)

        if (not labprinter.busy()) and (not printQueue.isEmpty()):
            nexttask = printQueue.dequeue()
            waitingtimes.append(nexttask.waitTime(currentSecond))
            labprinter.start_next(nexttask)

        labprinter.tick()

    averange_wait = sum(waitingtimes) / len(waitingtimes)
    print('Average Wait {:8.2f} secs {:2d} tasks remaining.'.format(averange_wait, printQueue.size()))


def newPrintTask():
    num = random.randrange(1, 181)
    if num == 180:
        return True
    else:
        return False


if __name__ == '__main__':
    for _ in range(10):
        simulation(3600, 5)
