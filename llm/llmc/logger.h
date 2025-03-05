// Logs to the output directory and uses the append mode

#ifndef LOGGER_H
#define LOGGER_H

#include <assert.h>
#include <stdio.h>
#include <string.h>

// defines: fopenCheck, freadCheck, fcloseCheck etc
#include "utils.h"

typedef struct
{
    int active;
    char output_log_file[512]; // prob the length of each line
} Logger;

void logger_init(Logger *logger, const char *log_dir, int process_rank, int resume)
{
    if (log_dir != NULL && process_rank == 0)
    { // running on a gpu, first gpu would have rank 0
        logger->active = 1;
        assert(strlen(log_dir) < 500);
        snprintf(logger->output_log_file, 512, "%s/main.log", log_dir);
        if (resume == 0)
        {
            // wipe existing log file if we're starting afresh
            FILE *logfile = fopenCheck(logger->output_log_file, "w"); // in write mode, it doesn't append
            fclose(logfile);
        }
    }
}

void logger_log_eval(Logger *logger, int step, float val)
{
    if (logger->active == 1)
    {
        // Append to logfile if active
        FILE *logfile = fopenCheck(logger->output_log_file, "a");
        fprintf(logfile, "s:%d eval: %.4f\n", step, val);
        fclose(logfile);
    }
}

void logger_log_val(Logger *logger, int step, float val_loss)
{
    if (logger->active == 1)
    {
        FILE *logfile = fopenCheck(logger->output_log_file, "a");
        fprintf(logfile, "s:%d tel:%.4f\n", step, val_loss);
        fclose(logfile);
    }
}

void logger_log_train(Logger *logger, int step, float train_loss, float lr, float grad_norm)
{
    if (logger->active == 1)
    {
        FILE *logfile = fopenCheck(logger->output_log_file, "a");
        fprintf(logfile, "s:%d trl:%.4f lr:%.6f norm:%.2f\n", step, train_loss, lr, grad_norm);
        fclose(logfile); // open file, write to it, close file
    }
}

#endif