package com.github.redstonevalley.arttree.jcuda;

import java.util.logging.Level;
import java.util.logging.Logger;

import com.github.redstonevalley.arttree.jcuda.kernels.Product;

import jcuda.LogLevel;
import jcuda.driver.JCudaDriver;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaComputeMode;
import jcuda.runtime.cudaDeviceProp;
import jcuda.utils.KernelLauncher;

public class TreeArtJcudaApp {

  public static final int IMAGE_WIDTH = 1024;
  public static final int IMAGE_HEIGHT = 1024;
  private static final cudaDeviceProp DESIRED_DEVICE_PROPERTIES = new cudaDeviceProp();

  static {
    JCuda.initialize();
    DESIRED_DEVICE_PROPERTIES.computeMode = cudaComputeMode.cudaComputeModeDefault;
  }

  public static CudaPitchedArray newTreeNodeArray(CudaDevice device) {
    return CudaPitchedArray.create(IMAGE_WIDTH, IMAGE_HEIGHT, device);
  }

  public static void main(String[] args) {
    Logger.getLogger(KernelLauncher.class.getName()).setLevel(Level.ALL);
    JCudaDriver.setLogLevel(LogLevel.LOG_DEBUGTRACE);
    JCuda.setLogLevel(LogLevel.LOG_DEBUGTRACE);
    CudaDevice device = CudaDevice.find(DESIRED_DEVICE_PROPERTIES);
    System.out.format("Using this device: %s\n", device);
    CudaPitchedArray x = newTreeNodeArray(device);
    CudaPitchedArray y = newTreeNodeArray(device);
    System.out.format("x: %s\n" + "y: %s\n", x, y);
    Product productKernel = new Product();
    CudaPitchedArray output = productKernel.execute(device, x, y);
    System.out.format("Result: %s\n", output);
  }

}
