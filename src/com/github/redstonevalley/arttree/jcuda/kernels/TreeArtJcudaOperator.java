package com.github.redstonevalley.arttree.jcuda.kernels;

import com.github.redstonevalley.arttree.jcuda.CudaDevice;
import com.github.redstonevalley.arttree.jcuda.CudaPitchedArray;
import com.github.redstonevalley.arttree.jcuda.TreeArtJcudaApp;

import jcuda.Pointer;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaStream_t;
import jcuda.runtime.dim3;
import jcuda.utils.KernelLauncher;

public abstract class TreeArtJcudaOperator {

  protected final KernelLauncher kernelLauncher;
  protected final String functionName;
  protected final cudaStream_t stream = new cudaStream_t();

  protected final dim3 gridDim = new dim3();
  protected final dim3 blockDim = new dim3();
  protected final int sharedMem = 0;// TreeArtJcudaApp.IMAGE_WIDTH * TreeArtJcudaApp.IMAGE_HEIGHT * 4;

  protected TreeArtJcudaOperator(String kernelSource, String functionName) {
    JCuda.cudaStreamCreate(stream);
    this.functionName = functionName;
    kernelLauncher = KernelLauncher.compile(
        kernelSource,
        functionName,
        "-I\"C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\"",
        "-L\"C:\\Program Files (x86)\\Windows Kits\\10\\Lib\\10.0.10240.0\\um\\x64\"");
    kernelLauncher.setSharedMemSize(sharedMem).setGridSize(32, 1).setBlockSize(32, 1, 1);
  }

  public CudaPitchedArray execute(CudaDevice device, CudaPitchedArray... inputs) {
    CudaPitchedArray output = TreeArtJcudaApp.newTreeNodeArray(device);
    Object[] args = new Object[4 + inputs.length];
    for (int i=0; i < inputs.length; i++) {
      args[i] = Pointer.to(inputs[i]);
    }
    args[args.length - 4] = Pointer.to(output);
    args[args.length - 3] = output.getWidth();
    args[args.length - 2] = output.getHeight();
    args[args.length - 1] = output.getPitch();
    
    kernelLauncher.call(args);
    return output;
  }
}
