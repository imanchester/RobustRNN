import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.utils.clip_grad as clip_grad
import time


def is_legal(v):
    legal = not torch.isnan(v).any() and not torch.isinf(v)
    return legal


def make_default_options(max_epochs=100, lr=1E-3, lr_decay=0.95, mu0=10, patience=20, clip_at=0.5, mu_rate=1.5, mu_max=1E6, alpha=0.5):
    options = {"max_epochs": max_epochs, "lr": lr, "debug": False, "mu0": mu0, "mu_rate": mu_rate, "mu_max": mu_max,
               "patience": patience, "lr_decay": lr_decay, "clip_at": clip_at, "alpha": alpha
               }
    return options


class stochastic_nlsdp():
    # decVars should be a list of parameter vectors and ceq cineq should be lists of functions
    # def __init__(self, model, train_loader, val_loader, criterion=None, equ=None, max_epochs=1000, lr=1.0, max_ls=50,
    #              tolerance_grad=1E-6, tolerance_change=1E-6, tolerance_constraint=1E-6, debug=False,
    #              patience=10, omega0=1E-2, eta0=1E-1, mu0=10, lr_decay=0.95, clip_at=1.0):

    # decVars should be a list of parameter vectors and ceq cineq should be lists of functions
    def __init__(self, model, train_loader, val_loader, criterion=None, max_epochs=1000, lr=1.0, max_ls=50,
                 tolerance_change=1E-5, debug=False, patience=10, mu0=10, lr_decay=0.95, clip_at=1.0,
                 alpha=0.5, mu_rate=1.5, mu_max=1E6):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Add model parameters to list of decision variables
        self.decVars = list(self.model.parameters())

        self.criterion = criterion
        self.patience = patience
        self.lr = lr
        self.lr_decay = lr_decay
        self.max_ls = max_ls

        self.alpha = alpha

        self.mu0 = mu0
        self.mu_rate = mu_rate
        self.mu_max = mu_max
        self.clip_at = clip_at

        self.max_epochs = max_epochs
        self.tolerance_change = tolerance_change

        self.LMIs = []
        self.inequality = []

        self.regularizers = []

    # Evaluates the equality constraints c_i(x) as a vector c(x)
    def ceq(self):
        if self.equConstraints.__len__() == 0:
            return None

        views = []
        for c in self.equConstraints:
            views.append(c())

        return torch.cat(views, 0)

    # returns a flattened tensor of all parameters
    def flatten_params(self):
        views = []
        for p in self.decVars:
            if p is None:
                view = p.new(p.numel()).zero_()
            elif p.is_sparse:
                view = p.to_dense().view(-1)
            else:
                view = p.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    # Returns the gradients as a flattened tensor
    def flatten_grad(self):
        views = []
        for p in self.decVars:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    # Adds the SDP constraint Qf > 0. Qf should be a function that returns the LMI
    # so that we want Qf()>0
    def addSDPconstraint(self, Qf):
        self.LMIs += [Qf]

    def addInequality(self, ineq):
        self.inequality += [ineq]

    def eval_LMIs(self):
        if self.LMIs.__len__() == 0:
            return None

        else:
            LMIs = []
            for lmi in self.LMIs:
                LMIs.append(lmi())

            return LMIs

    def eval_ineq(self):
        # Evaluates the inequality constraints and returns as a list.
        if self.inequality.__len__() == 0:
            return None

        else:
            inequalities = []
            for ineq in self.inequality:
                inequalities += ineq()

            return inequalities

    def checkLMIs(self):
        lbs = []
        for lmi in self.LMIs:
            min_eval = lmi().eig()[0]
            lbs += [min_eval[0].min()]

        return lbs

    # Adds a regualizers reg where reg returns term that we woule like to regularize by
    def add_regularizer(self, reg):
        self.regularizers += [reg]

    # Evaluates the regularizers
    def eval_regularizers(self):

        if self.regularizers.__len__() == 0:
            return None

        res = 0
        for reg in self.regularizers:
            res += reg()

        return res

    #  eta tol is the desired tolerance in the constraint satisfaction
    #  omega_tol is the tolerance for first order optimality
    def solve(self):
        # linsearch parameter

        def validate(loader):
            total_loss = 0.0
            total_batches = 0

            self.model.eval()
            with torch.no_grad():
                for idx, u, y in loader:
                    yest = self.model(u)
                    # total_loss += self.criterion(yest, y) * u.size(0)
                    error = yest - y
                    total_loss = torch.sqrt(torch.mean(error ** 2)) / torch.sqrt(torch.mean(y**2))
                    total_batches += u.size(0)

            return float(total_loss / total_batches)

        # Initial Parameters
        muk = self.mu0  # barrier parameter

        no_decrease_counter = 0

        with torch.no_grad():
            vloss = validate(self.val_loader)
            tloss = validate(self.train_loader)

            best_loss = vloss
            best_model = self.model.clone()

        log = {"val": [vloss], "training": [tloss], "epoch": [0], "muk": [muk]}
        optimizer = torch.optim.Adam(params=self.decVars, lr=self.lr)
        # optimizer = torch.optim.SGD(params=self.decVars, lr=self.lr)

        #  Main Loop of Optimizer
        for epoch in range(self.max_epochs):
            start_time = time.time()

            #  --------------- Training Step ---------------
            train_loss = 0.0
            total_batches = 0
            self.model.train()
            for (idx, u, y) in self.train_loader:

                def AugmentedLagrangian():
                    optimizer.zero_grad()

                    h0 = self.model.h0[idx, :]
                    yest = self.model(u, h0)
                    # yest = self.model(u)
                    L = self.criterion(y, yest)

                    error = yest.detach().numpy() - y.detach().numpy()
                    train_loss = np.sqrt(np.mean(error**2)) / np.sqrt(np.mean(y.detach().numpy()**2))

                    # train_loss = float(L) * u.size(0)

                    reg = self.eval_regularizers()
                    if reg is not None:
                        L += reg

                    barrier = 0

                    # Add barrier functions for inequality constraints
                    inequalities = self.eval_ineq()
                    if inequalities is not None:
                        for ineq in inequalities:
                            barrier += -ineq.log() / muk

                    # Add barrier functions for LMI constraints
                    LMIs = self.eval_LMIs()
                    if LMIs is not None:
                        for lmi in LMIs:
                            barrier += -lmi.logdet() / muk

                            try:
                                _ = torch.cholesky(lmi)  # try a cholesky factorization to ensure positive definiteness
                            except:
                                barrier = torch.tensor(float("inf"))

                    L += barrier

                    L.backward()

                    # n_params = sum(p.numel() for p in self.model.parameters())
                    clip_grad.clip_grad_norm_(self.model.parameters(), self.clip_at, "inf")
                    # max_grad = max([torch.norm(p, "inf") for p in self.model.parameters()])
                    g = [p.grad.abs().max() for p in filter(lambda p: p.grad is not None, self.model.parameters())]

                    return L, train_loss, barrier, max(g)

                def check_constraints():
                    # evaluates just the barrier function.
                    with torch.no_grad():
                        barrier = torch.Tensor([0.0])
                        # Add barrier functions for inequality constraints
                        inequalities = self.eval_ineq()
                        if inequalities is not None:
                            for ineq in inequalities:
                                barrier += -ineq.log() / muk

                        # Add barrier functions for LMI constraints
                        LMIs = self.eval_LMIs()
                        if LMIs is not None:
                            for lmi in LMIs:
                                barrier += -lmi.logdet() / muk

                                try:
                                    _ = torch.cholesky(lmi)  # try a cholesky factorization to ensure positive definiteness
                                except:
                                    barrier = torch.tensor(float("inf"))

                    return barrier

                # Store old parameters
                old_theta = self.model.flatten_params().detach()

                # step model
                Lag, t_loss, barrier, max_grad = optimizer.step(AugmentedLagrangian)
                new_theta = self.model.flatten_params().detach()

                # Perform a backtracking linesearch to avoid inf or NaNs
                barrier = check_constraints()
                ls = 0
                while not is_legal(barrier):

                    # step back by half
                    new_theta = self.alpha * old_theta + (1 - self.alpha) * new_theta
                    self.model.write_flat_params(new_theta)

                    ls += 1
                    if ls == 100:

                        print("maximum ls reached")
                        log["success"] = False
                        return log, best_model

                    barrier = check_constraints()

                train_loss += t_loss
                total_batches += u.size(0)


                print("Epoch {:4d}: \t[{:03d}],\tlr: {:1.1e},\t loss: {:.4f} ls: {:d},\tbarrier parameter: {:.1f}, |g| {:f}".format(epoch,
                      total_batches + 1, optimizer.param_groups[0]["lr"], train_loss / total_batches, ls, muk, max_grad))

            print("Time = ", time.time() - start_time)
            # ---------------- Validation Step ---------------
            vloss = validate(self.val_loader)
            # tloss = validate(self.train_loader)
            tloss = 0.0

            if vloss < best_loss - self.tolerance_change:
                no_decrease_counter = 0
                best_loss = vloss
                best_model = self.model.clone()

            else:
                no_decrease_counter += 1

            log["val"] += [vloss]
            log["training"] += [tloss]
            log["epoch"] += [epoch]
            log["muk"] += [muk]


            print("-" * 120)
            print("Epoch {:4d}\t train_loss {:.4f},\tval_loss: {:.4f},\tbarrier parameter: {:.4f}".format(epoch, tloss, vloss, muk))
            print("-" * 120)

            torch.save(self.model.state_dict(), './results/temp_model.params')

            if no_decrease_counter > self.patience:
                no_decrease_counter = 0
                # Reduce weight on barrier functions
                muk = self.mu_rate * muk
                if muk > self.mu_max:
                    break

                # Reduce learning rate slightly after each epoch
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= self.lr_decay

            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = self.lr / np.sqrt(float(epoch + 1))

        log["success"] = True
        return log, best_model


class stochastic_nlsdp_ee():
    # decVars should be a list of parameter vectors and ceq cineq should be lists of functions
    # def __init__(self, model, train_loader, val_loader, criterion=None, equ=None, max_epochs=1000, lr=1.0, max_ls=50,
    #              tolerance_grad=1E-6, tolerance_change=1E-6, tolerance_constraint=1E-6, debug=False,
    #              patience=10, omega0=1E-2, eta0=1E-1, mu0=10, lr_decay=0.95, clip_at=1.0):

    # decVars should be a list of parameter vectors and ceq cineq should be lists of functions
    def __init__(self, model, train_loader, val_loader, criterion=None, max_epochs=1000, lr=1.0, max_ls=50,
                 tolerance_change=1E-6, debug=False, patience=10, mu0=10, lr_decay=0.95, clip_at=1.0,
                 alpha=0.5, mu_rate=1.5, mu_max=1E6):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Add model parameters to list of decision variables
        self.decVars = list(self.model.parameters())

        self.criterion = criterion
        self.patience = patience
        self.lr = lr
        self.lr_decay = lr_decay
        self.max_ls = max_ls

        self.alpha = alpha

        self.mu0 = mu0
        self.mu_rate = mu_rate
        self.mu_max = mu_max
        self.clip_at = clip_at

        self.max_epochs = max_epochs
        self.tolerance_change = tolerance_change

        self.LMIs = []
        self.inequality = []

        self.regularizers = []

    # Evaluates the equality constraints c_i(x) as a vector c(x)
    def ceq(self):
        if self.equConstraints.__len__() == 0:
            return None

        views = []
        for c in self.equConstraints:
            views.append(c())

        return torch.cat(views, 0)

    # returns a flattened tensor of all parameters
    def flatten_params(self):
        views = []
        for p in self.decVars:
            if p is None:
                view = p.new(p.numel()).zero_()
            elif p.is_sparse:
                view = p.to_dense().view(-1)
            else:
                view = p.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    # Returns the gradients as a flattened tensor
    def flatten_grad(self):
        views = []
        for p in self.decVars:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    # Adds the SDP constraint Qf > 0. Qf should be a function that returns the LMI
    # so that we want Qf()>0
    def addSDPconstraint(self, Qf):
        self.LMIs += [Qf]

    def addInequality(self, ineq):
        self.inequality += [ineq]

    def eval_LMIs(self):
        if self.LMIs.__len__() == 0:
            return None

        else:
            LMIs = []
            for lmi in self.LMIs:
                LMIs.append(lmi())

            return LMIs

    def eval_ineq(self):
        # Evaluates the inequality constraints and returns as a list.
        if self.inequality.__len__() == 0:
            return None

        else:
            inequalities = []
            for ineq in self.inequality:
                inequalities += ineq()

            return inequalities

    def checkLMIs(self):
        lbs = []
        for lmi in self.LMIs:
            min_eval = lmi().eig()[0]
            lbs += [min_eval[0].min()]

        return lbs

    # Adds a regualizers reg where reg returns term that we woule like to regularize by
    def add_regularizer(self, reg):
        self.regularizers += [reg]

    # Evaluates the regularizers
    def eval_regularizers(self):

        if self.regularizers.__len__() == 0:
            return None

        res = 0
        for reg in self.regularizers:
            res += reg()

        return res

    #  eta tol is the desired tolerance in the constraint satisfaction
    #  omega_tol is the tolerance for first order optimality
    def solve(self):
        # linsearch parameter

        def validate(loader, ee=False):
            total_loss = 0.0
            total_batches = 0

            self.model.eval()
            with torch.no_grad():
                for idx, u, y in loader:
                    if ee:
                        yest = self.model.one_step(u)
                    else:
                        yest = self.model(u)

                    total_loss += self.criterion(yest, y) * y.size(0)
                    total_batches += y.size(0)

            return float(np.sqrt(total_loss / total_batches))

        # Initial Parameters
        # muk = self.mu0  # barrier parameter
        muk = self.mu0  # barrier parameter

        no_decrease_counter = 0

        with torch.no_grad():
            vloss = validate(self.val_loader)
            tloss = validate(self.train_loader, ee=True)

            best_loss = vloss
            best_model = self.model.clone()

        log = {"val": [vloss], "training": [tloss], "epoch": [0], "muk": [muk]}
        optimizer = torch.optim.Adam(params=self.decVars, lr=self.lr)
        # optimizer = torch.optim.SGD(params=self.decVars, lr=self.lr)

        #  Main Loop of Optimizer
        for epoch in range(self.max_epochs):

            #  --------------- Training Step ---------------
            train_loss = 0.0
            total_batches = 0
            self.model.train()
            for (idx, u, y) in self.train_loader:

                def AugmentedLagrangian():
                    optimizer.zero_grad()

                    yest = self.model.one_step(u)
                    # yest = self.model(u)
                    L = self.criterion(y, yest)

                    # train_loss = float(L) * y.size(0)
                    error = yest.detach().numpy() - y.detach().numpy()
                    train_loss = np.sqrt(np.mean(error**2)) / np.sqrt(np.mean(y.detach().numpy()**2))

                    reg = self.eval_regularizers()
                    if reg is not None:
                        L += reg

                    barrier = 0

                    # Add barrier functions for inequality constraints
                    inequalities = self.eval_ineq()
                    if inequalities is not None:
                        for ineq in inequalities:
                            barrier += -ineq.log() / muk

                    # Add barrier functions for LMI constraints
                    LMIs = self.eval_LMIs()
                    if LMIs is not None:
                        for lmi in LMIs:
                            barrier += -lmi.logdet() / muk

                            try:
                                _ = torch.cholesky(lmi)  # try a cholesky factorization to ensure positive definiteness
                            except:
                                barrier = torch.tensor(float("inf"))

                    L += barrier

                    L.backward()

                    # n_params = sum(p.numel() for p in self.model.parameters())
                    clip_grad.clip_grad_norm_(self.model.parameters(), self.clip_at, "inf")
                    # max_grad = max([torch.norm(p, "inf") for p in self.model.parameters()])
                    g = [p.grad.abs().max() for p in filter(lambda p: p.grad is not None, self.model.parameters())]

                    return L, train_loss, barrier, max(g)

                def check_constraints():
                    # evaluates just the barrier function.
                    with torch.no_grad():
                        barrier = torch.Tensor([0.0])
                        # Add barrier functions for inequality constraints
                        inequalities = self.eval_ineq()
                        if inequalities is not None:
                            for ineq in inequalities:
                                barrier += -ineq.log() / muk

                        # Add barrier functions for LMI constraints
                        LMIs = self.eval_LMIs()
                        if LMIs is not None:
                            for lmi in LMIs:
                                barrier += -lmi.logdet() / muk

                                try:
                                    _ = torch.cholesky(lmi)  # try a cholesky factorization to ensure positive definiteness
                                except:
                                    barrier = torch.tensor(float("inf"))

                    return barrier

                # Store old parameters
                old_theta = self.model.flatten_params().detach()

                # step model
                Lag, t_loss, barrier, max_grad = optimizer.step(AugmentedLagrangian)
                new_theta = self.model.flatten_params().detach()

                # Perform a backtracking linesearch to avoid inf or NaNs
                barrier = check_constraints()
                ls = 0
                while not is_legal(barrier):

                    # step back by half
                    new_theta = self.alpha * old_theta + (1 - self.alpha) * new_theta
                    self.model.write_flat_params(new_theta)

                    ls += 1
                    if ls == 100:

                        print("maximum ls reached")
                        log["success"] = False
                        return log, best_model

                    barrier = check_constraints()

                train_loss += t_loss
                total_batches += y.size(0)

                print("Epoch {:4d}: \t[{:03d}],\tlr: {:1.1e},\t loss: {:.4f} ls: {:d},\tbarrier parameter: {:.1f}, |g| {:f}".format(epoch,
                      total_batches + 1, optimizer.param_groups[0]["lr"], train_loss / total_batches, ls, muk, max_grad))

            # ---------------- Validation Step ---------------
            vloss = validate(self.val_loader, ee=False)
            tloss = validate(self.train_loader, ee=True)

            if vloss < best_loss - self.tolerance_change:
                no_decrease_counter = 0
                best_loss = vloss
                best_model = self.model.clone()

            else:
                no_decrease_counter += 1

            log["val"] += [vloss]
            log["training"] += [tloss]
            log["epoch"] += [epoch]
            log["muk"] += [muk]

            print("-" * 120)
            print("Epoch {:4d}\t train_loss {:.4f},\tval_loss: {:.4f},\tbarrier parameter: {:.4f}".format(epoch, tloss, vloss, muk))
            print("-" * 120)

            torch.save(self.model.state_dict(), './results/temp_model.params')

            if no_decrease_counter > self.patience:
                no_decrease_counter = 0
                # Reduce weight on barrier functions
                muk = self.mu_rate * muk
                if muk > self.mu_max:
                    break

                # Reduce learning rate slightly after each epoch
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= self.lr_decay

            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = self.lr / np.sqrt(float(epoch + 1))

        log["success"] = True
        return log, best_model
