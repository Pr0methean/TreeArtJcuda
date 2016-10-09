package com.github.redstonevalley.arttree.jcuda;

import jcuda.NativePointerObject;
import jcuda.Pointer;
import jcuda.runtime.JCuda;

public class CudaPitchedArray extends NativePointerObject {

  protected static final int ELEMENT_SIZE = 4;

  protected final long pitch;
  protected final long width;
  protected final long height;

  public static CudaPitchedArray create(long width, long height, CudaDevice device) {
    Pointer pointer = new Pointer();
    long[] pitch = new long[1];
    device.setAsCurrent();
    JCuda.cudaMallocPitch(pointer, pitch, width * ELEMENT_SIZE, height);
    return new CudaPitchedArray(pointer, pitch[0], width, height);
  }

  public long getPitch() {
    return pitch;
  }

  public long getWidth() {
    return width;
  }

  public long getHeight() {
    return height;
  }

  public int getElementSize() {
    return ELEMENT_SIZE;
  }

  protected CudaPitchedArray(Pointer pointer, long pitch, long width, long height) {
    super(pointer);
    this.pitch = pitch;
    this.width = width;
    this.height = height;
  }
  
  @Override
  protected void finalize() {
    JCuda.cudaFree(Pointer.to(this));
  }
}
