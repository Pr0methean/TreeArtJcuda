package com.github.redstonevalley.arttree.jcuda;

import jcuda.runtime.JCuda;
import jcuda.runtime.cudaDeviceProp;

/**
 * A CUDA device that will be used to run modules if it is the most recent one for which the
 * calling thread has called {@link #setAsCurrent()}.
 *
 * @author cryoc
 */
public class CudaDevice {

  @Override
  public String toString() {
    return String.format("CudaDevice [id=%d, properties=%s]", id, properties);
  }

  private final int id;

  private final cudaDeviceProp properties;

  static {
    JCuda.setExceptionsEnabled(true);
  }

  private CudaDevice(int id, cudaDeviceProp properties) {
    this.id = id;
    this.properties = properties;
  }

  public void setAsCurrent() {
    JCuda.cudaSetDevice(id);
  }

  public static CudaDevice find(cudaDeviceProp properties) {
    int[] device = new int[1];
    JCuda.cudaChooseDevice(device, properties);
    JCuda.cudaSetDevice(device[0]);
    JCuda.cudaGetDeviceProperties(properties, device[0]);
    return new CudaDevice(device[0], properties);    
  }

  public int getId() {
    return id;
  }

  public cudaDeviceProp getProperties() {
    return properties;
  }

}
