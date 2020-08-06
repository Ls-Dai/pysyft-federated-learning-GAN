import torch.nn.functional as func

def train_unit(config, model, device, federated_train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(federated_train_loader):
        # send model to federated points
        model.send(data.location)
        data, target = data.to(device), target.to(device)
        # gradient zeroed
        optimizer.zero_grad()
        output = model(data)
        # loss calculation
        loss = func.nll_loss(output, target)
        # gradient calculation
        loss.backward()
        optimizer.step()
        # update model from sever
        model.get()
        if batch_idx % config.log_interval == 0:
            loss = loss.get()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * config.batch_size, len(federated_train_loader) * config.batch_size,
                       100. * batch_idx / len(federated_train_loader), loss.item()))

def train(config, model, device, federated_train_loader, optimizer):
    for epoch in range(1, config.epochs + 1):
        train_unit(config, model, device, federated_train_loader, optimizer, epoch)