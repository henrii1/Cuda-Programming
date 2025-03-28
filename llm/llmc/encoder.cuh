/*
The GPT-2 Encoder, which combines two encodings: token and position
In the forward pass, both encodings are added together
In the backward pass, the gradients flow to both, handled by different kernels
*/

#ifndef ENCODER_CUH
#define ENCODER_CUH

#include <assert.h>
#include <stdint.h>
#include <utility>  //std::pair
#include <vector>
#include <algorithm>
#include <unordered_map>

#include "cuda_common.h"
#include "cuda_utils.cuh"