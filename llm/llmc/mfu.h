// get_flops_promised. prob an optimization utility

#ifndef MFU_H
#define MFU_H

#include <stdio.h>
#include <stdlib.h> //malloc, free, exit, system
#include <string.h>

#if __has_include(<nvml.h>) // nvidia management library header

#define USE_NVML 1
#include <nvml.h>
#else
#define USE_NVML 0
#endif

// using define to rep enum
#define MFUH_PRECISION_FP32 0
#define MFUH_PRECISION_FP16 1
#define MFUH_PRECISION_BF16 2

#if USE_NVML // when 1
inline void nvml_check(nvmlReturn_t status, const char *file, int line)
{
    if (status != NVML_SUCCESS)
    {
        printf("[NVML ERROR] at file %s:%d:\n%s\n", file, line, nvmlErrorString(status));
        exit(EXIT_FAILURE);
    }
}; // because it is expanded to be inline, like a macro but doesn't actuall need it

#define nvmlCheck(err) (nvml_check(err, __FILE__, __LINE__)) // basically run the printf function.
#endif

typedef struct
{
    float TF_32;    // tensor-core performance 32 bit
    float BF_16_32; //  bf16 with 32 bit accumulate (mixed precision, matmul in 16, weight accum in 32)
    float FP_16_32; //  fp16 with 32 bit accumulate
    float FP_16_16; // fp16 with 16 bit accumulate
    float FP_8_32;  //
    float FP_8_16;
    float CLOCK; // clock frequency
    float CORES;
} PerfData;

// basic default data from the nvidia whitepapers
static const PerfData VOLTA = {125.0f, -1.f, 125.f, -1.f, -1.f, -1.f, 1530.f, 640.f};
static const PerfData AMPERE_DATACENTER = {156.f, 312.f, 312.f, 312.f, -1.f, -1.f, 1410.f, 432.f};
static const PerfData AMPERE_CONSUMER = {40.f, 80.f, 80.f, 160.f, -1.f, -1.f, 1860.f, 336.f};
static const PerfData HOPPER = {378.f, 756.f, 756.f, 756.f, 1513.f, 1513.f, 1620.f, 456.f};
static const PerfData ADA = {82.6f, 165.2f, 165.2f, 330.3f, 330.3f, 660.6f, 2520.f, 512.f};

typedef struct
{
    const char *name;
    const PerfData *perf_data;
    float new_cores;
    float new_mhz;
} GPUEntry;

// the overrides for each specific GPU
// an array of GPUEntry
static GPUEntry gpu_db[] = {
    {"Tesla V100-SXM2-16GB", &VOLTA, 640, 1530},
    {"Tesla V100-PCIE-32GB", &VOLTA, 640, 1530},
    {"NVIDIA A100-PCIE-40GB", &AMPERE_DATACENTER, 432, 1410},
    {"NVIDIA A100-PCIE-80GB", &AMPERE_DATACENTER, 432, 1410},
    {"NVIDIA A100-SXM4-40GB", &AMPERE_DATACENTER, 432, 1410},
    {"NVIDIA A100-SXM4-80GB", &AMPERE_DATACENTER, 432, 1410},
    {"NVIDIA RTX A2000", &AMPERE_CONSUMER, 104, 1200},
    {"NVIDIA RTX A4000", &AMPERE_CONSUMER, 192, 1560},
    {"NVIDIA RTX A4500", &AMPERE_CONSUMER, 224, 1650},
    {"NVIDIA RTX A5000", &AMPERE_CONSUMER, 256, 1695},
    {"NVIDIA RTX A5500", &AMPERE_CONSUMER, 320, 1770},
    {"NVIDIA RTX A6000", &AMPERE_CONSUMER, 336, 1800},
    {"NVIDIA GeForce RTX 3090 Ti", &AMPERE_CONSUMER, 336, 1860},
    {"NVIDIA GeForce RTX 3090", &AMPERE_CONSUMER, 328, 1695},
    {"NVIDIA GeForce RTX 3080 Ti", &AMPERE_CONSUMER, 320, 1665},
    {"NVIDIA GeForce RTX 3080", &AMPERE_CONSUMER, 272, 1710},
    {"NVIDIA GeForce RTX 3070 Ti", &AMPERE_CONSUMER, 192, 1770},
    {"NVIDIA GeForce RTX 3070", &AMPERE_CONSUMER, 184, 1725},
    {"NVIDIA GeForce RTX 3060 Ti", &AMPERE_CONSUMER, 152, 1665},
    {"NVIDIA GeForce RTX 3060", &AMPERE_CONSUMER, 112, 1777},
    {"NVIDIA RTX A2000 ADA", &ADA, 88, 2130},
    {"NVIDIA RTX A4000 ADA", &ADA, 192, 2175},
    {"NVIDIA RTX A4500 ADA", &ADA, 224, 2580},
    {"NVIDIA RTX A5000 ADA", &ADA, 400, 2550},
    {"NVIDIA RTX A5880 ADA", &ADA, 440, 2460},
    {"NVIDIA RTX A6000 ADA", &ADA, 568, 2505},
    {"NVIDIA GeForce RTX 4090", &ADA, 512, 2520},
    {"NVIDIA GeForce RTX 4080 SUPER", &ADA, 320, 2550},
    {"NVIDIA GeForce RTX 4080", &ADA, 304, 2505},
    {"NVIDIA GeForce RTX 4070 Ti SUPER", &ADA, 264, 2610},
    {"NVIDIA GeForce RTX 4070 Ti", &ADA, 240, 2610},
    {"NVIDIA GeForce RTX 4070 SUPER", &ADA, 224, 2475},
    {"NVIDIA GeForce RTX 4070", &ADA, 184, 2475},
    {"NVIDIA GeForce RTX 4070", &ADA, 184, 2475},
    {"NVIDIA GeForce RTX 4060 Ti", &ADA, 136, 2535},
    {"NVIDIA GeForce RTX 4060", &ADA, 96, 2460},
    {"NVIDIA H100 PCIe", &HOPPER, 456, 1620},
    {"NVIDIA H100 80GB HBM3", &HOPPER, 528, 1830}, // HBM3 = SXM5
};

float get_flops_promised(const char *device, int precision_mode)
{

    if (!(precision_mode == MFUH_PRECISION_FP32 || precision_mode == MFUH_PRECISION_FP16 || precision_mode == MFUH_PRECISION_BF16))
    {
        fprintf(stderr, "Invalid precision mode: %d", precision_mode);
        return -1.0f
    }

    // linearly search until you find your gpu, then calculate flops promised
    int num_gpu_entries = sizeof(gpu_db);
    for (int i = 0; i < num_gpu_entries; i++)
    {
        if (strcmp(gpu_db[i].name, device) == 0)
        {
            const PerfData *perf_data = gpu_db.perf_data;

            // look up flop value for the given precision mode
            float value = -1.0f;
            if (precision_mode == MFUH_PRECISION_BF16)
            {
                value = perf_data->BF_16_32;
            }
            if (precision_mode == MFUH_PRECISION_FP32)
            {
                value = perf_data->TF_32;
            }
            if (precision_mode == MFUH_PRECISION_FP16)
            {
                value = perf_data->FP_16_32;
            }

            // for gpus without bf16 capabilities
            if (value < 0.0f)
            {
                fprintf(stderr, "No data for GPU %s and precision mode %d \n", device, precision_mode);
                return -1.0f;
            }
            // adjust flops based on core count of gpu
            float new_cores = gpu_db[i].new_cores;
            float new_mhz = gpu_db[i].new_mhz;
            float adjusted = value * (new_cores / perf_data->CORES) * (new_mhz / perf_data->CLOCK);
            return adjusted;
        }
    }
    return -1.0f;
}

struct GPUUtilInfo{
    unsigned int clock;
    unsigned int max_clock;
    unsigned int power;
    unsigned int power_limit;
    unsigned int fan;
    unsigned int temperature;
    unsigned int temp_slowdown;

    float gpu_utilization;
    float mem_utilization;
    const char* throttle_reason;
};

// Initialize nvml and generate a handle to GPU

#if USE_NVML
nvmlDevice_t nvml_get_device(){
    static bool needs_init = true;
    static nvmlDevice_t device;
    if (needs_init){
        needs_init = false;
        nvmlCheck(nvmlInit());
        nvmlCheck(nvmlDeviceGetHandleByIndex_v2(0, &device));
    }
    return device;
}

//convert bitfield into a text reason.
const char* get_throttle_reason(unsigned long long bits){
    if (bits & (nvmlClocksThrottleReasonSwPowerCap | nvmlClocksThrottleReasonHwPowerBrakeSlowdown)){
        return "power cap";
    } else if (bits & (nvmlClocksThrottleReasonSwThermalSlowdown | nvmlClocksThrottleReasonHwThermalSlowdown)){
        return "thermal cap";
    } else if (bits & (nvmlClocksThrottleReasonAll)){
        return "other cap";
    } else {
        return "no cap";
    }
}

// gather data for a GPUUtilInfo object
GPUUtilInfo get_gpu_utilization_info() {
    GPUUtilInfo info;
    nvmlDevice_t device = nvml_get_device();
    // query different infos directly
    nvmlCheck(nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &info.clock));
    nvmlCheck(nvmlDeviceGetMaxClockInfo(device, NVML_CLOCK_SM, &info.max_clock));
    nvmlCheck(nvmlDeviceGetPowerManagementLimit(device, &info.power_limit));
    nvmlCheck(nvmlDeviceGetPowerUsage(device, &info.power));
    nvmlCheck(nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &info.temperature));
    nvmlCheck(nvmlDeviceGetTemperatureThreshold(device, NVML_TEMPERATURE_THRESHOLD_SLOWDOWN, &info.temp_slowdown));
    unsigned long long throttle;
    nvmlCheck(nvmlDeviceGetCurrentClocksThrottleReasons(device, &throttle));
    info.throttle_reason = get_throttle_reason(throttle);
    nvmlCheck(nvmlDeviceGetFanSpeed(device, &info.fan));

    // for "utilization", we look at recorded samples. In principle, we could query the driver for how many samples
    // to request, but then we'd need to dynamically allocate sufficient space. Let's just hard-code a limit of 128,
    // and have no memory management required
    constexpr const int BUFFER_LIMIT = 128;
    nvmlSample_t buffer[BUFFER_LIMIT];
    nvmlValueType_t v_type;
    unsigned int sample_count = BUFFER_LIMIT;
    nvmlCheck(nvmlDeviceGetSamples(device, NVML_GPU_UTILIZATION_SAMPLES, 0, &v_type, &sample_count, buffer));
    float gpu_utilization = 0.f;
    for(unsigned i = 0; i < sample_count; ++i) {
        gpu_utilization += (float)buffer[i].sampleValue.uiVal;
    }
    gpu_utilization /= (float)sample_count;

    // sample count may have been modified by the query above; reset back to buffer size
    sample_count = BUFFER_LIMIT;
    nvmlCheck(nvmlDeviceGetSamples(device, NVML_MEMORY_UTILIZATION_SAMPLES, 0, &v_type, &sample_count, buffer));
    float mem_utilization = 0.f;
    for(unsigned i = 0; i < sample_count; ++i) {
        mem_utilization += (float)buffer[i].sampleValue.uiVal;
    }
    mem_utilization /= (float)sample_count;

    info.gpu_utilization = gpu_utilization;
    info.mem_utilization = mem_utilization;
    return info;
}

#else

GPUUtilInfo get_gpu_utilization_info() {
    fprintf(stderr, "Error: Compiled without nvml support. Cannot perform additional GPU state tracking.");
    exit(EXIT_FAILURE);
}

#endif
#endif