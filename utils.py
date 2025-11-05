import torch
from torch import nn
import time
import os
import numpy as np
# from torch.func import jacrev, vmap
from pyhessian import hessian
# from pytorchcv.model_provider import get_model as ptcv_get_model # model
# from new_hessian import power_iteration
from models import get_model
from get_scheduler import get_scheduler
from tqdm import tqdm


class Full_Batch_Experiment:
    def __init__(self, opt):
        # self.data = data.to(device)
        # self.labels = labels.to(device)
        self.total_loss = 0
        self.history = {
            'loss': [],
            'train_acc': [],
            'val_acc': [],
            'test_loss': []
        }
        self.grads = []
        self.const_loss = None
        self.name = None
        # self.gpu_ids = opt.gpu
        # self.train_sampler = opt.train_sampler

    def train(self, train_loader, val_loader, num_layers, nodes, initial_lr, scheduler_name, model_num, epochs=100, init_weights="normal", device='cuda', our_lr=None):
        # device = self.gpu_ids
        if scheduler_name == "ours":
            lr_given = True
            if our_lr is None:
                print("calculating LR from scratch!")
                lr_given = False
        self.name = scheduler_name
        size = 0
        
        # self.train_sampler.set_epoch(0)
        for img, label in train_loader:
            img = img.view(img.shape[0], -1)
            size = img.shape[1]
            break
        
        self.model = get_model(model_num, size, num_layers, nodes, init_weights=init_weights)
        # self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.model.cuda()
        # self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.gpu_ids])

        self.lr = initial_lr
        if model_num != 6:
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.loss_fn = nn.L1Loss()

        self.epochs = epochs
        self.lipschitz = None
        if scheduler_name == "ours":
            if lr_given:
                self.lr = our_lr
            else:
            # self.model = ptcv_get_model("resnet20_cifar10", pretrained=True).to(device)
                self.model.eval()
                self.const_loss = 0
                self.lipschitz = self.check_lipshitz(train_loader, self.epochs, num_layers, nodes, model_num, init_weights)
                # self.train_sampler.set_epoch(0)
                for img, label in train_loader:
                    img = img.to(device)
                    label = label.to(device)
                    if model_num == 0 or model_num==6:
                        img = img.view(img.shape[0], -1)
                    # self.optimizer.zero_grad()
                    with torch.no_grad():
                        logits = self.model(img)
                    if model_num!=6:
                        loss = self.loss_fn(logits, label)
                    else:
                        loss = self.loss_fn(logits, img)

                    self.const_loss += loss.item()

                    print(f"initial loss for model using 'our' scheduler is : {self.const_loss}")
                assert self.lipschitz is not None
                self.lr = np.sqrt((2*self.const_loss) / (self.lipschitz * self.epochs * len(train_loader)))
            print(f"our LR: {self.lr}")

        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
        if epochs == 20:
            step_num=10
        elif epochs == 50:
            step_num=20
        else:
            step_num=40

        self.scheduler = get_scheduler(scheduler_name, self.optimizer, self.lr, step_num, initial_lr, len(train_loader), epochs)
        start_time = time.time()
        print(scheduler_name)
        for epoch in range(self.epochs):
            total_correct = 0
            self.model.train()
            self.total_loss = 0
            grad_norm = 0
            # self.train_sampler.set_epoch(epoch)
            for img, label in train_loader:
                img = img.to(device)
                label = label.to(device)
                if model_num == 0 or model_num ==6:
                    img = img.view(img.shape[0], -1)
                self.optimizer.zero_grad()
                logits = self.model(img)
                if model_num != 6:
                    loss = self.loss_fn(logits, label)
                    pred = logits.argmax(dim=1)
                    total_correct += (pred == label).sum().item()
                else:
                    loss = self.loss_fn(logits, img)
                    total_correct = 0
                self.total_loss += loss.item()
                grad_norm += self.grad_norm_calc(loss, self.model)
                loss.backward()
                self.optimizer.step()
                if scheduler_name == 'cyclic':
                    self.scheduler.step()

            grad_norm = np.sqrt(grad_norm / len(train_loader))
            train_acc = total_correct / len(train_loader.dataset)
            self.grads.append(grad_norm)
            self.total_loss /= len(train_loader)
            self.history['loss'].append(self.total_loss)
            self.history['train_acc'].append(train_acc)
            if (epoch+1) % 20 == 0:
                print(f"{epoch+1}/{self.epochs} --> grad_norm : {grad_norm}")

            total_correct = 0
            test_loss = 0
            self.model.eval()
            with torch.no_grad():
                for img, label in val_loader:
                    img = img.to(device)
                    label = label.to(device)
                    if model_num == 0 or model_num==6:
                        img = img.view(img.shape[0], -1)
                    logits = self.model(img)
                    if model_num != 6:
                        loss = self.loss_fn(logits, label)
                        pred = logits.argmax(dim=1)
                        total_correct += (pred == label).sum().item()
                    else:
                        loss = self.loss_fn(logits, img)
                        total_correct = 0
                    test_loss += loss.item()
                val_acc = total_correct / len(val_loader.dataset)
                self.history["val_acc"].append(val_acc)
                test_loss /= len(val_loader)
                self.history['test_loss'].append(test_loss)

            if scheduler_name == 'ours':
                self.scheduler.step(self.const_loss, self.epochs, self.lipschitz)
            # elif scheduler_name == 'plateau':
            #     self.scheduler.step(self.total_loss)
            elif scheduler_name == 'cyclic':
                pass
            else:
                self.scheduler.step()
            if (epoch+1) % 20 == 0:
                # print(self.optimizer)
                time_taken = time.time() - start_time
                print(f"{epoch+1}/{self.epochs} --> loss : {self.total_loss} | time taken : {time_taken}")
                start_time = time.time()

    def grad_norm_calc(self, loss, model):
        grad_norm = torch.zeros(1, requires_grad=True).to('cuda')
        grads = torch.autograd.grad(loss, model.parameters(), retain_graph=True)
        for grad in grads:
            grad_norm = grad_norm + torch.sum(grad ** 2)
        # grad_norm = torch.sqrt(grad_norm)
        return grad_norm.item()

    def check_lipshitz(self, train_loader, num_iters, num_layers, nodes, model_num, init_weights, device='cuda'):
        print("calculating lipschitz")
        # device = gpu_ids
        lipschitz = 0
        size = 0
        # self.train_sampler.set_epoch(0)
        for img, label in train_loader:
            img = img.view(img.shape[0], -1)
            size = img.shape[1]
            break
        t = time.time()
       
        for i in range(num_iters):
            model = get_model(model_num, size, num_layers, nodes, init_weights=init_weights)
            # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model.cuda()
            # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu_ids])

            if model_num!=6:
                criterion = nn.CrossEntropyLoss()
            else:
                criterion = nn.L1Loss()

            # self.train_sampler.set_epoch(i)
            for img, label in train_loader:
                if model_num == 0 or model_num ==6:
                    img = img.view(img.shape[0], -1)
                img = img.to(device)
                label = label.to(device)
                if model_num ==6:
                    flag=True
                else:
                    flag=False
                hessian_comp = hessian(model, criterion, data=(img, label), cuda=True, flag=flag)

            top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues()
            
            lipschitz = max(lipschitz, top_eigenvalues[-1])
            print(top_eigenvalues[-1])

        print("lipschitz: ", lipschitz)
        print(f"time taken: {time.time()-t}")
        del model
        return lipschitz


class Mini_Batch_Experiment:
    def __init__(self, opt):
        # self.data = data.to(device)
        # self.labels = labels.to(device)
        self.total_loss = 0
        self.history = {
            'loss': [],
            'train_acc': [],
            'val_acc': [],
            'test_loss': []
        }
        self.grads = []
        self.const_loss = None
        self.name = None
        # self.gpu_ids = opt.gpu
        # self.train_sampler = opt.train_sampler

    def train(self, train_loader, val_loader, num_layers, nodes, initial_lr, scheduler_name, model_num, epochs=100, init_weights="normal", device='cuda', our_lr=None):
        # device = self.gpu_ids
        if scheduler_name == "ours":
            lr_given = True
            if our_lr is None:
                print("calculating LR from scratch!")
                lr_given = False
            
        self.name = scheduler_name
        size = 0
        # self.train_sampler.set_epoch(0)
        for img, label in train_loader:
            img = img.view(img.shape[0], -1)
            size = img.shape[1]
            break
        if epochs == 20:
            step_num=10
        elif epochs == 50:
            step_num=20
        else:
            step_num=40

        self.model = get_model(model_num, size, num_layers, nodes, init_weights=init_weights)
        # self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.model.cuda()
        # self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.gpu_ids])
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"trainable params: {total_params}")

        self.lr = initial_lr
        self.loss_fn = nn.CrossEntropyLoss()
        self.epochs = epochs
        self.lipschitz = None
        if scheduler_name == "ours":
            if lr_given:
                self.lr =  our_lr
            else: 
                self.model.eval()
                self.const_loss = 0
                if epochs >=100:
                    self.lipschitz = 40.11 #self.check_lipshitz(val_loader, self.epochs//2, num_layers, nodes, model_num, init_weights)
                else:
                    self.lipschitz = 40.11 #self.check_lipshitz(val_loader, self.epochs, num_layers, nodes, model_num, init_weights)

                # self.train_sampler.set_epoch(0)
                for img, label in tqdm(train_loader, desc='Training'):
                    img = img.to(device)
                    label = label.to(device)
                    if model_num == 0:
                        img = img.view(img.shape[0], -1)
                    # self.optimizer.zero_grad()
                    with torch.no_grad():
                        logits = self.model(img)
                    loss = self.loss_fn(logits, label)
                    self.const_loss += loss.item()
                    # print(loss.item())

                self.const_loss /= len(train_loader)
                print(f"initial loss for model using 'our' scheduler is : {self.const_loss}")
                assert self.lipschitz is not None
                self.lr =  np.sqrt((2*self.const_loss) / (self.lipschitz*self.epochs*len(train_loader))) # epochs * total_dataset / batch_size
                # print(self.epochs*len(train_loader))
            print(f"our LR: {self.lr}")
            # return 

        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
        # add optim

        self.scheduler = get_scheduler(scheduler_name, self.optimizer, self.lr, step_num, initial_lr, len(train_loader), epochs)

        start_time = time.time()
        print(scheduler_name)
        for epoch in range(self.epochs):
            total_correct = 0
            self.model.train()
            self.total_loss = 0
            grad_norm = 0
            # self.train_sampler.set_epoch(epoch)
            for img, label in tqdm(train_loader, desc='Training'):
                img = img.to(device)
                label = label.to(device)
                if model_num == 0:
                    img = img.view(img.shape[0], -1)
                self.optimizer.zero_grad()
                logits = self.model(img)
                loss = self.loss_fn(logits, label)
                pred = logits.argmax(dim=1)
                total_correct += (pred == label).sum().item()
                self.total_loss += loss.item()
                grad_norm += self.grad_norm_calc(loss, self.model)
                loss.backward()
                self.optimizer.step()

                if scheduler_name == 'cyclic':
                    self.scheduler.step()

            grad_norm = np.sqrt(grad_norm / len(train_loader))
            train_acc = total_correct / len(train_loader.dataset)
            self.grads.append(grad_norm)
            self.total_loss /= len(train_loader)
            self.history['loss'].append(self.total_loss)
            self.history['train_acc'].append(train_acc)

            if (epoch+1) % 5 == 0:
                print(f"{epoch+1}/{self.epochs} --> grad_norm : {grad_norm}")

            total_correct = 0
            test_loss = 0
            self.model.eval()
            with torch.no_grad():
                for img, label in val_loader:
                    img = img.to(device)
                    label = label.to(device)
                    if model_num == 0:
                        img = img.view(img.shape[0], -1)
                    logits = self.model(img)
                    loss = self.loss_fn(logits, label)
                    test_loss += loss.item()
                    pred = logits.argmax(dim=1)
                    total_correct += (pred == label).sum().item()
                val_acc = total_correct / len(val_loader.dataset)
                self.history["val_acc"].append(val_acc)
                test_loss /= len(val_loader)
                self.history['test_loss'].append(test_loss)


            if scheduler_name == 'ours':
                self.scheduler.step(self.const_loss, self.epochs, self.lipschitz)
            # elif scheduler_name == 'plateau':
            #     self.scheduler.step(self.total_loss)
            elif scheduler_name == 'cyclic':
                pass
            else:
                self.scheduler.step()
            if (epoch+1) % 5 == 0:
                # print(self.optimizer)
                time_taken = time.time() - start_time
                print(f"{epoch+1}/{self.epochs} --> loss : {self.total_loss} | time taken : {time_taken}")
                # with open('./temp_res', 'a') as f:
                #     f.write(str(epoch) + "\n")
                #     f.write(str(self.total_loss) + "\n")
                #     f.write(str(grad_norm) + "\n")
                start_time = time.time()

    def grad_norm_calc(self, loss, model):
        grad_norm = torch.zeros(1, requires_grad=True).to('cuda')
        grads = torch.autograd.grad(loss, model.parameters(), retain_graph=True)
        for grad in grads:
            grad_norm = grad_norm + torch.sum(grad ** 2)
        # grad_norm = torch.sqrt(grad_norm)
        return grad_norm.item()

    def check_lipshitz(self, train_loader, num_iters, num_layers, nodes, model_num, init_weights, device='cuda'):
        print("calculating lipschitz")
        # device = gpu_ids
        lipschitz = 0
        size = 0
        # self.train_sampler.set_epoch(0)
        for img, label in train_loader:
            img = img.view(img.shape[0], -1)
            size = img.shape[1]
            break
        t = time.time()
        for i_ in range(num_iters):
            model = get_model(model_num, size, num_layers, nodes, init_weights=init_weights)
            # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model.cuda()
            # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu_ids])

            criterion = nn.CrossEntropyLoss()
            grad_norm = torch.zeros(1, requires_grad=True).to(device)
            max_for_batch = 0
            # self.train_sampler.set_epoch(i_)
            for img, label in tqdm(train_loader, desc="lipschitz"):
                if model_num == 0:
                    img = img.view(img.shape[0], -1)
                img = img.to(device)
                label = label.to(device)
                hessian_comp = hessian(model, criterion, data=(img, label), cuda=True)
                top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues()
                # print("The top Hessian eigenvalue of this model is %.4f" % top_eigenvalues[-1])
                max_for_batch = max(max_for_batch, top_eigenvalues[-1])
                # top_eigenvalues = power_iteration(model, criterion, img, label, num_iters=100, tol=1e-6)
                # max_for_batch = max(max_for_batch, top_eigenvalues)
                # print(max_for_batch)

            lipschitz = max(lipschitz, max_for_batch)
            print(lipschitz)

        print("lipschitz:", lipschitz)
        print(f"time taken: {time.time() - t}")
        del model
        return lipschitz


class Step_Size_Experiment:
    def __init__(self, ours=True, opt=None):
        # self.data = data.to(device)
        # self.labels = labels.to(device)
        self.ours = ours
        self.total_loss = 0
        self.history = {
            'loss': [],
            'train_acc': [],
            'val_acc': [],
            'test_loss': []
        }
        self.grads = []
        self.const_loss = None
        self.name = None
        # self.gpu_ids = opt.gpu
        # self.train_sampler = opt.train_sampler

    def train(self, train_loader, val_loader, num_layers, nodes, initial_lr, model_num, init_weights="normal", epochs=100, device='cuda', our_lr=None):
        # device = self.gpu_ids
        if self.ours:
            lr_given = True
            if our_lr is None:
                print("calculating LR from scratch!")
                lr_given = False
        size = 0
        # self.train_sampler.set_epoch(0)
        for img, label in train_loader:
            img = img.view(img.shape[0], -1)
            size = img.shape[1]
            break

        self.model = get_model(model_num, size, num_layers, nodes, init_weights=init_weights)
        # self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.model.cuda()
        # self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.gpu_ids])

        self.lr = initial_lr
        self.epochs = epochs
        self.loss_fn = nn.CrossEntropyLoss()
        self.lipschitz = None

        if self.ours:
            if lr_given:
                self.lr = our_lr
            else: 
                self.model.eval()
                self.const_loss = 0
                # self.lipschitz = self.check_lipshitz(train_loader, self.epochs//3, num_layers, nodes, model_num, init_weights)  #train_loader, num_iters, num_layers, nodes
                # # self.train_sampler.set_epoch(0)
                # for img, label in tqdm(train_loader, desc='Evaluating'):
                #     img = img.to(device)
                #     label = label.to(device)
                #     if model_num == 0:
                #         img = img.view(img.shape[0], -1)
                #     # self.optimizer.zero_grad()
                #     with torch.no_grad():
                #         logits = self.model(img)
                #     loss = self.loss_fn(logits, label)
                #     self.const_loss += loss.item()

                # self.const_loss /= len(train_loader)
                # print(f"initial loss for model using 'our' scheduler is : {self.const_loss}")
                # assert self.lipschitz is not None
                self.lr = 0.003491 #np.sqrt((2*self.const_loss) / (self.lipschitz * self.epochs * len(train_loader))) 
                print(self.lr)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
        self.scheduler = None
        if self.ours:
            step_num = None
            self.scheduler = get_scheduler("ours", self.optimizer, self.lr, step_num, initial_lr, len(train_loader), epochs)

        start_time = time.time()

        if self.ours:
            print(f"ours lr: {self.lr}")
        else:
            print(f"working on: {self.lr}")

        for epoch in range(self.epochs):
            total_correct = 0
            self.model.train()
            self.total_loss = 0
            grad_norm = 0
            # self.train_sampler.set_epoch(epoch)
            for img, label in tqdm(train_loader, desc='Training'):
                img = img.to(device)
                label = label.to(device)
                if model_num == 0:
                    img = img.view(img.shape[0], -1)
                self.optimizer.zero_grad()
                logits = self.model(img)
                loss = self.loss_fn(logits, label)
                pred = logits.argmax(dim=1)
                total_correct += (pred == label).sum().item()
                self.total_loss += loss.item()
                grad_norm += self.grad_norm_calc(loss, self.model)
                loss.backward()
                self.optimizer.step()

            grad_norm = np.sqrt(grad_norm / len(train_loader))
            train_acc = total_correct / len(train_loader.dataset)
            self.grads.append(grad_norm)
            self.total_loss /= len(train_loader)
            self.history['loss'].append(self.total_loss)
            self.history['train_acc'].append(train_acc)

            if epoch+1 % 20 == 0:
                print(f"{epoch+1}/{self.epochs} --> grad_norm : {grad_norm}")

            total_correct = 0
            test_loss = 0
            self.model.eval()
            with torch.no_grad():
                for img, label in tqdm(val_loader, desc='Testing'):
                    img = img.to(device)
                    label = label.to(device)
                    if model_num == 0:
                        img = img.view(img.shape[0], -1)
                    logits = self.model(img)
                    loss = self.loss_fn(logits, label)
                    test_loss += loss.item()
                    pred = logits.argmax(dim=1)
                    total_correct += (pred == label).sum().item()
                val_acc = total_correct / len(val_loader.dataset)
                self.history["val_acc"].append(val_acc)
                test_loss /= len(val_loader)
                self.history['test_loss'].append(test_loss)

            if self.ours:
                self.scheduler.step(self.const_loss, self.epochs, self.lipschitz)

            if (epoch+1) % 20 == 0:
                # print(self.optimizer)
                time_taken = time.time() - start_time
                print(f"{epoch+1}/{self.epochs} --> loss : {self.total_loss} | time taken : {time_taken}")
                start_time = time.time()

    def grad_norm_calc(self, loss, model):
        grad_norm = torch.zeros(1, requires_grad=True).to('cuda')
        grads = torch.autograd.grad(loss, model.parameters(), retain_graph=True)
        for grad in grads:
            grad_norm = grad_norm + torch.sum(grad ** 2)
        # grad_norm = torch.sqrt(grad_norm)
        return grad_norm.item()

    def check_lipshitz(self, train_loader, num_iters, num_layers, nodes, model_num, init_weights, device='cuda', d=None):
        print("calculating lipschitz")
        # device = gpu_ids
        lipschitz = 0
        size = 0
        # self.train_sampler.set_epoch(0)
        for img, label in train_loader:
            img = img.view(img.shape[0], -1)
            size = img.shape[1]
            break
        t = time.time()
        # output_dim = len(torch.unique(torch.tensor(train_loader.dataset.targets)))
        for i in range(num_iters):
            # print(f"{i} / {num_iters}")
            model = get_model(model_num, size, num_layers, nodes, init_weights=init_weights)
            # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model.cuda()
            # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu_ids])

            criterion = nn.CrossEntropyLoss()
            #grad_norm = torch.zeros(1, requires_grad=True).to(device)
            max_for_batch = 0
            # self.train_sampler.set_epoch(i)
            for img, label in tqdm(train_loader, desc='Lipschitz'):
                if model_num == 0:
                    img = img.view(img.shape[0], -1)
                img = img.to(device)
                label = label.to(device)

                # output = model(img)
                # loss = criterion(output, label)
                hessian_comp = hessian(model, criterion, data=(img, label), cuda=True)
                top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues()
                max_for_batch = max(max_for_batch, top_eigenvalues[-1])
                print(max_for_batch)

            # print("The top Hessian eigenvalue of this model is %.4f" % top_eigenvalues[-1])
            lipschitz = max(lipschitz, max_for_batch)
            print(f"cur lip: {lipschitz}")

        print("lipschitz:", lipschitz)
        print(f"time taken: {time.time() - t}")
        del model
        return lipschitz
