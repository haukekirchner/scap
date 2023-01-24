[...]
for epoch in range(num_training_epochs):
    [...]    
    prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(logdir + "/profiler"),
            record_shapes=True,
            profile_memory=True,
            with_stack=True)
    prof.start()
    for i, batch in enumerate(train_loader):
        if i >= (1 + 1 + 3) * 2:
                    break
        [...] # Load data, classify data, calculate loss
        loss.backward(); optimizer.step()
        prof.step()