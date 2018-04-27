import subprocess
import time

class Scenario :
    def __init__(self,STARTTIME):
        self.isStressRunChecker = [0 for i in range(5000)]
        self.isEndFlag = False
        self.framerate = 10
        self.starttime = STARTTIME
        self.end = 140

    def getCurrentFramerate(self):
        return self.framerate

    def isEnd(self):
        now = time.time() - self.starttime
        if now > self.end:
            self.isEndFlag = True

        return self.isEndFlag


    def playScenario(self):
        now = time.time() - self.starttime

        if now < 20:
            self.framerate = 5

        elif now < 40:
            self.framerate = 13


        elif now < 60:
            self.framerate = 2

        elif now < 80:
            self.framerate = 9

        elif now < 100:
            if self.isStressRunChecker[1468] == 0:
                self.isStressRunChecker[1468] = 1
                for i in range(3) :
                    subprocess.Popen('glmark2 -b refract:show-fps=true:duration=15', shell=True)



        elif now > self.end:

            self.framerate = 1
            self.isEndFlag = True

        '''
        
        elif now < 120:
            self.framerate = 5

        elif now < 140:
            self.framerate = 5

            if self.isStressRunChecker[23] == 0:
                self.isStressRunChecker[23] = 1
                for i in range(5) :
                    subprocess.Popen('glmark2 -b refract:show-fps=true:duration=15', shell=True)
        
        '''

