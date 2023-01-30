class data_init:
    x_train = 0
    x_test = 0
    x_val = 0
    y_train = 0
    y_test = 0
    y_val = 0
    
    def m_set_total(self, x_train, x_test, y_train, y_test):
        self.set_x_train(x_train)
        self.set_x_test(x_test)
        self.set_y_train(y_train)
        self.set_y_test(y_test)
    
    def d_set_total(self, x_train, x_test, y_train, y_test, x_val, y_val):
        self.set_x_train(x_train)
        self.set_x_test(x_test)
        self.set_y_train(y_train)
        self.set_y_test(y_test)
        self.set_x_val(x_val)
        self.set_y_val(y_val)
    
    def set_x_train(self, x_train):
        self.x_train=x_train
        
    def set_x_test(self, x_test):
        self.x_test=x_test
        
    def set_x_val(self, x_val):
        self.x_val=x_val
    
    def set_y_train(self, y_train):
        self.y_train=y_train
    
    def set_y_test(self, y_test):
        self.y_test=y_test
        
    def set_y_val(self, y_val):
        self.y_val=y_val
        
    def get_x_train(self):
        return self.x_train
        
    def get_x_test(self):
        return self.x_test
    
    def get_x_val(self):
        return self.x_val
    
    def get_y_train(self):
        return self.y_train
    
    def get_y_test(self):
        return self.y_test
    
    def get_y_val(self):
        return self.y_val