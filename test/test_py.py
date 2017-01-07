

class A:
    def __init__(self, pid):
        self.pid = pid

    def ok(self):
        print self.pid
        self.ok2()

    def ok2(self):
        print "A"
    
class B(A):
    def __init__(self, pid):
        self.pid = pid

    def ok2(self):
        print "B"

b = B(3)
b.ok()