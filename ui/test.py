class C():
    def f(self):
        def f1(msg):
            print(f'f1: {msg}')
        print('In f')
        f1('a')
    def g():
        print('In g')
        
c = C()
c.f()        
c.f().f1('b')