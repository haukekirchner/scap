from deepspeed.profiling.flops_profiler import FlopsProfiler
[...]
flop_prof = FlopsProfiler(model)
profile_step = 5; print_profile= True
for epoch in range(num_training_epochs):
    [...]
    for i, batch in enumerate(train_loader):
            if i == profile_step:
                flop_prof.start_profile()
            [...] # Load data, classify data, calculate loss
            if i == profile_step: # end profiling and print output
                flop_prof.stop_profile()
                flops = flop_prof.get_total_flops()
                macs = flop_prof.get_total_macs()
                params = flop_prof.get_total_params()
                if print_profile:
                    flop_prof.print_model_profile(profile_step=profile_step)
                flop_prof.end_profile()
            loss.backward(); optimizer.step()