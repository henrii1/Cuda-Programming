// scheduler header

#ifndef SCHEDULERS_H
#define SCHEDULERS_H

#include <assert.h>
#include <math.h>
#include <string.h>

typedef struct
{
    const char *type;
    float learning_rate;
    int warmup_iteration;
    int train_num_batches;
    float final_learning_rate_frac;
} LearningRateScheduler;

void lr_scheduler_init(LearningRateScheduler *scheduler, char *type, float lr, int warmup_iters,
                       int train_num_batches, float final_learning_rate_frac)
{
    scheduler->type = type;
    scheduler->learning_rate = lr;
    scheduler->warmup_iteration = warmup_iters;
    scheduler->train_num_batches = train_num_batches;
    scheduler->final_learning_rate_frac = final_learning_rate_frac;
}

float get_learning_rate_cosine(LearningRateScheduler *scheduler, int step)
{
    float lr = scheduler->learning_rate;
    if (step < scheduler->warmup_iteration)
    {
        lr = scheduler->learning_rate * ((float)(step + 1)) / scheduler->warmup_iteration;
    }
    else
    {
        float decay_ratio = ((float)(scheduler->warmup_iteration)) / (scheduler->train_num_batches - scheduler->warmup_iteration);
        assert(0.0f <= decay_ratio && decay_ratio < 1.0f);
        float coeff = 0.5f * (1.0f + cosf(M_PI * decay_ratio));
        assert(0.0f <= coeff && coeff < 1.0f);
        float min_lr = scheduler->learning_rate * scheduler->final_learning_rate_frac;
        lr = min_lr + coeff * (scheduler->learning_rate - min_lr);
    }
    return lr;
}

float get_learning_rate_linear(LearningRateScheduler *scheduler, int step)
{
    float lr = scheduler->learning_rate;
    if (step < scheduler->warmup_iteration)
    {
        lr = scheduler->learning_rate * ((float)(step + 1)) / (scheduler->warmup_iteration);
    }
    else
    {
        float decay_ratio = ((float)(step - scheduler->warmup_iteration)) / (scheduler->train_num_batches - scheduler->warmup_iteration);
        assert(0.0f <= decay_ratio && decay_ratio < 1.0f);
        float min_lr = scheduler->learning_rate * scheduler->final_learning_rate_frac;
        lr = scheduler->learning_rate - decay_ratio * (scheduler->learning_rate - min_lr);
    }
    return lr;
}

// get learning rate after being updated by above functions
float get_learning_rate_constant(LearningRateScheduler *scheduler, int step)
{
    return scheduler->learning_rate;
}

// learning rate scheduler wsd (from arxiv paper)
float get_learning_rate_wsd(LearningRateScheduler *scheduler, int step)
{
    int decay_point = (int)(0.8f * scheduler->train_num_batches);
    float max_lr = scheduler->learning_rate;
    float lr = max_lr;
    if (step < scheduler->warmup_iteration)
    {
        float decay_ratio = ((float)(step + 1)) / scheduler->warmup_iteration;
        lr = max_lr * decay_ratio;
    }
    else if (step < decay_point)
    {
        // keep lr constant
    }
    else
    {
        float decay_ratio = ((float)(step - decay_point)) / (scheduler->train_num_batches - decay_point);
        assert(0.0f <= decay_ratio && decay_ratio < 1.0f);
        float min_lr = max_lr * scheduler->final_learning_rate_frac;
        return min_lr + (1.0f - sqrtf(decay_ratio)) * (max_lr - min_lr);
    }

    return lr;
}

// return the lr at a given step
float get_learning_rate(LearningRateScheduler *scheduler, int step)
{
    float step_learning_rate;
    if (strcmp(scheduler->type, "cosine") == 0)
    { // string compare
        step_learning_rate = get_learning_rate_cosine(scheduler, step);
    }
    else if (strcmp(scheduler->type, "linear") == 0)
    {
        step_learning_rate = get_learning_rate_linear(scheduler, step);
    }
    else if (strcmp(scheduler->type, "constant") == 0)
    {
        step_learning_rate = get_learning_rate_constant(scheduler, step);
    }
    else if (strcmp(scheduler->type, "wsd") == 0)
    {
        step_learning_rate = get_learning_rate_wsd(scheduler, step);
    }
    else
    {
        fprintf(stderr, "Unknown learning rate scheduler type: %s\n", scheduler->type);
        exit(EXIT_FAILURE);
    }
    return step_learning_rate;
}

#endif