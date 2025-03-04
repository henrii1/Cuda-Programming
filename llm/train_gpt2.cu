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