// implementation of a simple Sampler

#ifndef SAMPLER_H
#define SAMPLER_H

#include <math.h>

// xorshift RNG
unsigned int random_u32(unsigned long long *state)
{
    // xorshift rng
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

float random_f32(unsigned long long *state)
{
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample_softmax(const float *logits, int n, float coin)
{
    // samples index from logits
    // coin is a random number in [0, 1]

    double norm = 0;
    for (int i = 0; i < n; i++)
    {
        norm += expf(logits[i]);
    }
    coin *= norm;
    float cdf = 0.0f;
    for (int i = 0; i < n; i++)
    {
        cdf += expf(logits[i]);

        if (coin < cdf)
        {
            return i;
        }
    }

    return n - 1; // last
}

#endif