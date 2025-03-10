/*
This outlier detector is one that we can use to monitor the loss and grad norm internally
by keeping track of a window of measurements, and for each time we add a measurement, it returns
the z-score of the new measurement w.r.t the window
*/

#include <stdio.h>
#include <math.h>

#define OUTLIER_DETECTOR_WINDOW_SIZE 128

typedef struct {
    double buffer[OUTLIER_DETECTOR_WINDOW_SIZE];
    int count;
    int index;
    double sum;
    double sum_sq;
} OutlierDetector;

void init_detector(OutlierDetector* detector){
    for (int i=0; i< OUTLIER_DETECTOR_WINDOW_SIZE; i++){
        detector->buffer[i] = 0.0;
    }
    detector->count = 0;
    detector->index = 0;
    detector->sum = 0.0;
    detector->sum_sq = 0.0;
}

double update_detector(OutlierDetector* detector, double new_value){
    if (detector->count < OUTLIER_DETECTOR_WINDOW_SIZE){
        detector->buffer[detector->count] = new_value;
        detector->sum += new_value;
        detector->sum_sq += new_value * new_value;
        detector->count++;
        return nan(""); // not enough data yet
    } else {
        // the window is filled, we can detect outliers now.

        // pop the oldest value from window
        double old_value = detector->buffer[detector->index]; //remember that index is not updated in if
        detector->sum -= old_value;
        detector->sum_sq -= old_value * old_value;
        // push new value to window
        detector->buffer[detector->index] = new_value;
        detector->sum += new_value;
        detector->sum_sq += new_value * new_value;
        //move index to the next position
        detector->index = (detector->index + 1) % OUTLIER_DETECTOR_WINDOW_SIZE;  // starts from zero again, after reaching the end
        //calculate the z-score of the new value
        double mean = detector->sum / OUTLIER_DETECTOR_WINDOW_SIZE;
        double variance = (detector->sum_sq / OUTLIER_DETECTOR_WINDOW_SIZE) - (mean * mean);
        double std_dev = sqrt(variance);
        if (std_dev == 0.0){
            return 0.0;
        }
        double z = (new_value - mean) / std_dev;
        return z;
    }
}