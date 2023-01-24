from torch.utils.tensorboard import SummaryWriter
[...]
loss_idx_value = 0
for epoch in range(num_training_epochs):
    model.train(); running_loss = 0.0   
    for i, batch in enumerate(train_loader):
        [...] # Load data, classify data, calculate loss
        loss.backward(); optimizer.step()
        running_loss += loss.item()
        writer.add_scalar("Loss/Minibatches", running_loss, loss_idx_value)
        loss_idx_value += 1
    writer.add_scalar("Loss/Epochs", running_loss, epoch)
    model.eval()
    if epoch % 5 == 4: # get validation accuracy every 5 epochs
        [...] # calculate accuracy
        writer.add_scalar("Accuracy", accuracy, epoch)