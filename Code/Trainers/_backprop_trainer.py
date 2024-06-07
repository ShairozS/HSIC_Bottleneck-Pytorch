class Backprop:

    
    def __init__(self, model, device, loss = "CE"):
        if args.model == "MLP":
            self.model  = MLP(args)
        if args.model == "signMLP":
            self.model  = signMLP(args)
        if args.model == "CNN":
            self.model  = CNN(args)
        if args.model == "VGG":
            self.model  = VGG(args)
        if args.model == 'KAN':
            self.model = MNISTChebyKAN(degree = 20)
        
        self.model.to(device)
        self.batch_size = args.batchsize
        self.lambda_0   = args.lambda_
        self.sigma      = args.sigma_
        self.last_linear = "output_layer"

        
        self.opt = optim.AdamW(self.model.parameters(), lr=0.001)
#         self.opt = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.01)
        self.iter_loss1, self.iter_loss2, self.iter_loss3 = [], [], []
        self.track_loss1, self.track_loss2, self.track_loss3 = [], [], []
        
        self.loss = args.loss
        if self.loss == "mse": self.output_criterion = nn.MSELoss()#y_pred, labels_float)
        elif self.loss == "CE": self.output_criterion = nn.CrossEntropyLoss()#y_pred, label)
        
    def step(self, input_data, labels):
        self.opt.zero_grad()

        labels_float = F.one_hot(labels, num_classes=10).float()
        
        y_pred, hidden_zs = self.model(input_data)

        if self.loss == "mse": 
            l = self.output_criterion(y_pred, labels_float)
        elif self.loss == "CE": 
            l = self.output_criterion(y_pred, labels)

        l.backward()
        self.opt.step()
        return(l)




class HSICBottleneck:
    def __init__(self, model, batchsize = 128, loss = 'CE', lambda_ = 100, sigma_ = 10, HSIC = 'nHSIC', kernel_x = 'rbf', kernel_y = 'student', kernel_h = 'rbf', forward = 'n', device='cuda'):
        
        self.model = model
        #if model == "MLP":
        #    self.model  = MLP(args)
        #if model == "signMLP":
        #    self.model  = signMLP(args)
        #if model == "CNN":
        #    self.model  = CNN(args)
        #if model == "VGG":
        #    self.model  = VGG(args)
        #if model == 'KAN':
        #    self.model = MNISTChebyKAN2(degree=8)
        
        self.batch_size = batchsize
        self.lambda_0   = lambda_
        self.sigma      = sigma_
        self.extractor  = 'hsic'
        self.last_linear = "output_layer"
        self.HSIC = compute_HSIC(HSIC)
        self.kernel = compute_kernel()
        self.kernel_x = kernel_x
        self.kernel_h = kernel_h
        self.kernel_y = kernel_y
        self.forward = forward
        self.device = device
        
        self.opt = optim.AdamW(self.model.parameters(), lr=0.001)
        self.iter_loss1, self.iter_loss2, self.iter_loss3 = [], [], []
        self.track_loss1, self.track_loss2, self.track_loss3 = [], [], []
        
        self.loss = loss
        if self.loss == "mse": self.output_criterion = nn.MSELoss()
        elif self.loss == "CE": self.output_criterion = nn.CrossEntropyLoss()
        
    def step(self, input_data, labels):
        
        labels_float = F.one_hot(labels, num_classes=10).float()
        if self.forward == "x": Kx  = self.kernel(input_data, self.sigma, self.kernel_x)
        Ky = self.kernel(labels_float, self.sigma, self.kernel_y)
        
        kernel_list = list()
        y_pred, hidden_zs = self.model(input_data)
        #print(y_pred.shape, [h.shape for h in hidden_zs])
        
        loss_LI = 0.
        for num, feature in enumerate(hidden_zs): 
            kernel_list.append(self.kernel(feature, self.sigma, self.kernel_h))
            
        
        total_loss1, total_loss2, total_loss3 = 0., 0., 0.
        for num, feature in enumerate(kernel_list):
            if num == (len(hidden_zs)-1): 
                if self.forward == "h": total_loss1 += self.HSIC(feature, kernel_list[num-1], self.batch_size, self.device)
                elif self.forward == "x": total_loss1 += self.HSIC(feature, Kx, self.batch_size, self.device)
                if self.loss == "mse": total_loss3 += self.output_criterion(hidden_zs[-1], labels_float)
                elif self.loss == "CE": total_loss3 += self.output_criterion(hidden_zs[-1], labels)
            elif num == 0:
                if self.forward == "x": total_loss1 += self.HSIC(feature, Kx, self.batch_size, self.device)
                total_loss2 += - self.lambda_0*self.HSIC(feature, Ky, self.batch_size, self.device)
            else:
                if self.forward == "h": total_loss1 += self.HSIC(feature, kernel_list[num-1], self.batch_size, self.device)
                elif self.forward == "x": total_loss1 += self.HSIC(feature, Kx, self.batch_size, self.device)
                total_loss2 += - self.lambda_0*self.HSIC(feature, Ky, self.batch_size, self.device)
        
        if self.forward == "h" or self.forward == "x": 
            total_loss = total_loss1 + total_loss2 + total_loss3 + loss_LI
            self.iter_loss1.append(total_loss1.item())
        if self.forward == "n": 
            total_loss = total_loss2 + total_loss3 + loss_LI
            self.iter_loss1.append(-1)
        self.opt.zero_grad()
        total_loss.backward()
        self.opt.step()
                
        self.iter_loss2.append(total_loss2.item())
        self.iter_loss3.append(total_loss3.item())
        
    def update_loss(self):
        self.track_loss1.append(np.mean(self.iter_loss1))
        self.track_loss2.append(np.mean(self.iter_loss2))
        self.track_loss3.append(np.mean(self.iter_loss3))
        self.iter_loss1, self.iter_loss2, self.iter_loss3 = [], [], []
    
    def tune_output(self, input_data, labels):
        self.model.train()
        if self.loss == "mse":
            one_hot_labels = F.one_hot(labels, num_classes=10)
            labels = F.one_hot(labels, num_classes=10).float()
        
        y_pred, hidden_zs = self.model(input_data)
        total_loss = self.output_criterion(hidden_zs[-1], labels)
        self.opt.zero_grad()
        total_loss.backward()
        self.opt.step()