// CUDA version of the code

#include <unistd.h> //read(), write(), chdir()
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <sys/types.h>

#include "llmc/utils.h" // defines the cpu fopenCheck, freadCheck, fcloseCheck

#include "llmc/tokenizer.h" // tokenizer_init, tokenizer_decode

#include "llmc/dataloader.h" // dataloader_init, evalloader_init

#include "llmc/rand.h" //defines manual_seed, normal_ (same as torch.normal)

#include "llmc/schedulers.h" // contains lr_scheduler_init, get_learning_rate

#include "llmc/sampler.h" // contains sample softmax and random_f32

#include "llmc/logger.h" // contains logger_init, logger_log_eval, ..._val, ..._train

#include "llmc/mfu.h" // contains get_flops_promised

#include "llmc/outlier_detector.h"  // contains OutlierDetector, init_detector, update_detector

/* GPU Utilities*/
#include "llmc/cuda_common.h" // contains WARP_SIZE, MAX_1024_THREADS_BLOCKS, CEIL_DIV, cudaCheck, PRECISION_MODE, NVTX_RANGE_FN

#include "llmc/cuda_utils.cuh" // packed128, f128, x128, warpReduceSum, warpReduceMax, blockReduce, copy_and_cast_kernel, cudaMallocConditionallyManaged.

#include "llmc/cublas_common.h"  //  CUBLAS_LOWP, cublasCheck, cublast_workspace_size, cublaslt_workspace cublas_compute, cublaslt_handle, cublas_handle

#include "llmc/encoder.cuh"  // encoder_forward, encoder_backward

#include "llmc/layernorm.cuh"  // layernorm_forward, residual_forward, fused_residual_forward5, layernorm_backward