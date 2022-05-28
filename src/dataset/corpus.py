class Foo:
    
    class Sub:
        def __init__(self, object) -> None:
            self.object = object
                
        def spam(self):
            print('I print a Foo attribute')
            print('Foo attribute is:', self.object.name)
            
        def set_name(self, name):
            self.object.name = name
            
            
    def __init__(self):
        self.name = 'foo'
        self.sub = Foo.Sub(self)
    
    def get_name(self):
        print(self.name)
    
        
inst = Foo()  
inst.sub.set_name('foo new')
inst.get_name()
