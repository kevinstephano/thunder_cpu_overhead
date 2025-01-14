import argparse
from collections import OrderedDict
from functools import partial, wraps
import subprocess
import sys
import thunder
import torch
from torch.profiler import profile, ProfilerActivity
import traceback
from typing import Callable

def install_pandas():
    """
    Check if pandas is installed, and if not, install it using pip.
    Returns True if pandas is successfully installed or already present,
    False if installation fails.
    """
    try:
        import pandas as pd
        return pd
    except ImportError:
        print("pandas is not installed. Installing now...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
            print("pandas has been successfully installed!")
            import pandas as pd
            return pd
        except subprocess.CalledProcessError:
            print("Failed to install pandas. Please try installing manually using:")
            print("pip install pandas")
            return None

def run(sys_argv, model_name, batch_size, sequence_length, model, input_fn, grad_fn=None) : 
    pd = install_pandas()
    assert pd is not None
    pd.options.display.max_colwidth=100
    pd.options.display.float_format = '{:.3f}'.format

    parser = argparse.ArgumentParser(description='Rope Examples')
    parser.add_argument('--nsys', default=False, action="store_true", help='Disables torch.profiler for nsys.')
    parser.add_argument('--warmup', default=5, type=int, help='Warmup iterations.')
    parser.add_argument('--iters', default=10, type=int, help='Timing iterations.')
    parser.add_argument('--execs', nargs='+', type=str, help='List of executor names to time.', default=["Torch-Eager", "torch.compile", "Thunder-torch.compile", "Thunder-Torch", "Thunder-nvFuser"], required=False)
    parser.add_argument('--thunder_trace', default=False, action="store_true", help='Prints a Thunder trace.')
    parser.add_argument('--nvfuser_repro', default=False, action="store_true", help='Prints an nvFuser reproduction script.')
    args,extra_args = parser.parse_known_args(args=sys_argv[1:])

    assert len(extra_args) == 0, "Unknown args: {}".format(extra_args)

    def eager_wrapper(model):
        return model

    executors = OrderedDict()
    for exec in args.execs:
        if exec == "Torch-Eager":
            executors["Torch-Eager"] = eager_wrapper
        elif exec == "Thunder-Torch":
            executors["Thunder-Torch"] = partial(thunder.jit, executors=["apex", "cudnn", "sdpa", "torch"])
        elif exec == "torch.compile":
            executors["torch.compile"] = partial(torch.compile)
        elif exec == "Thunder-nvFuser":
            executors["Thunder-nvFuser"] = partial(thunder.jit, executors=["cudnn", "sdpa", "nvfuser"], nv_enable_linear=True)
        elif exec == "Thunder-torch.compile":
            executors["Thunder-torch.compile"] = partial(thunder.jit, executors=["torchcompile"])
        else:
            assert False, f"Unknown executor: {exec}"

    benchmark_data = []
    for name, exec in executors.items():
        exec_model = exec(model)

        if ((name == "Thunder-nvFuser") or (name == "Thunder-Torch") or (name == "Thunder-torch.compile")) and args.thunder_trace:
            exec_model(**input_fn())
            print(name, "Forward:")
            fwd_trace = thunder.last_traces(exec_model)[-1]
            print(fwd_trace)
            
            if grad_fn is not None:
                print(name, "Backward:")
                bwd_trace = thunder.last_backward_traces(exec_model)[-1]
                print(bwd_trace)

        fd_fwd = {}
        fd_bwd = {}
        if name == "Thunder-nvFuser" and args.nvfuser_repro:
            exec_model(**input_fn())
            for key in thunder.last_traces(exec_model)[-1].python_ctx().keys():
                if key[0:8] == "nvFusion":
                    fd_fwd[key] = thunder.last_traces(exec_model)[-1].python_ctx()[key]
                    fd_fwd[key].store_inputs = True
            if grad_fn is not None:
                for key in thunder.last_backward_traces(exec_model)[-1].python_ctx().keys():
                    if key[0:8] == "nvFusion":
                        fd_bwd[key] = thunder.last_backward_traces(exec_model)[-1].python_ctx()[key]
                        fd_bwd[key].store_inputs = True

        def model_iter():
            torch.cuda.nvtx.range_push("Inputs Generation")
            input_dict = input_fn()
            torch.cuda.nvtx.range_pop()
            
            torch.cuda.nvtx.range_push("Forward")
            fwd_time = 0.0
            fwd_kernels = 0
            if not args.nsys:
                with profile(activities=[ProfilerActivity.CUDA]) as prof: 
                    y = exec_model(**input_dict)

                for evt in prof.events():
                    if evt.device_time > 0.0:
                        fwd_kernels += 1
                    fwd_time += evt.device_time
            else:
                y = exec_model(**input_dict)
            torch.cuda.nvtx.range_pop()
            
            bwd_time = 0.0
            bwd_kernels = 0
            if grad_fn is not None:
                torch.cuda.nvtx.range_push("Forward-Loss")
                for idx in range(0, len(y)):
                    if idx == 0:
                        loss = y[0]
                    else:
                        loss += y[idx]
                torch.cuda.nvtx.range_pop()
             
                torch.cuda.nvtx.range_push("Grad Generation")
                for param in exec_model.parameters():
                    param.grad = None
                grads = grad_fn()
                torch.cuda.nvtx.range_pop()
             
                torch.cuda.nvtx.range_push("Backward")
                if not args.nsys:
                    with profile(activities=[ProfilerActivity.CUDA]) as prof: 
                        loss.backward(grads)
                    for evt in prof.events():
                        if evt.device_time > 0.0:
                            bwd_kernels += 1
                        bwd_time += evt.device_time
                else:
                    loss.backward(grads)
                torch.cuda.nvtx.range_pop()

            return fwd_kernels, fwd_time, bwd_kernels, bwd_time


        def run_model():
            for _ in range(args.warmup):
                model_iter()
           
            fwd_kernel_time = 0
            bwd_kernel_time = 0
            fwd_kernels = 0
            bwd_kernels = 0
            for _ in range(args.iters):
                fwd_kernels, fwd_time, bwd_kernels, bwd_time = model_iter()
                fwd_kernel_time += fwd_time
                bwd_kernel_time += bwd_time

            fwd_kernel_time = fwd_kernel_time / args.iters / 1.e3
            bwd_kernel_time = bwd_kernel_time / args.iters / 1.e3
            
            return fwd_kernels, fwd_kernel_time, bwd_kernels, bwd_kernel_time

        fwd_time = 0.0
        bwd_time = 0.0
        fwd_kernels = 0
        bwd_kernels = 0
        torch.cuda.nvtx.range_push(f"Executor: {name}")
        try:
            fwd_kernels, fwd_time, bwd_kernels, bwd_time = run_model()
        except Exception as e:
            print(e)
            print(traceback.format_exc())
        torch.cuda.nvtx.range_pop()

        if name == "Thunder-nvFuser" and args.nvfuser_repro:
            for key in fd_fwd.keys():
                print(f"nvfuser Forward Repro: {key}")
                fd_fwd[key].last_used.execute(inputs=fd_fwd[key].last_inputs, print_repro=True)
            for key in fd_bwd.keys():
                print(f"nvfuser Backward Repro: {key}")
                fd_bwd[key].last_used.execute(inputs=fd_bwd[key].last_inputs, print_repro=True)

        benchmark_data.append([model_name, batch_size, sequence_length, name, fwd_kernels, fwd_time, bwd_kernels, bwd_time])
        #print(f"{model_name} {name} Fwd-Time: {fwd_time:.03f} ms Bwd-Time: {bwd_time:.03f} ms")

    df = pd.DataFrame(benchmark_data, columns=["Model", "Batch-Size", "Sequence-Length", "Executor", "Forward-Kernels", "Forward-Time(ms)", "Backward-Kernels", "Backward-Time(ms)"])
    print(df)

    return None
