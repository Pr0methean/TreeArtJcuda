package com.github.redstonevalley.arttree.jcuda.kernels;

public class Product extends TreeArtJcudaOperator {

  public Product() {
    super(/* @formatter:off */
          "extern \"C\"\n"
        + "__global__ void multiply(\n"
              + "float *a,\n"
              + "float *b,\n"
              + "float *output,\n"
              + "long width,\n"
              + "long height,\n"
              + "long pitch) {\n"
          + "long i = blockIdx.x * blockDim.x + threadIdx.x;\n"
          + "long n = width * height;\n"
          + "if (i < n) {"
            + "long row = i / width;\n"
            + "long col = i % width;\n"
            + "size_t offset = row * pitch + col;\n"
            + "output[offset] = a[offset] * b[offset];"
          + "}\n"
        + "}", /* @formatter:on */
        "multiply");
  }

}
