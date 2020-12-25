#include "math.h"
#include "Euclidean.h"



double Euclidean(double * a, double * b, int length)
{
  double ret = 0;
  for (int i=0; i<length;i++)
  {
    double dist = a[i]-b[i];
    ret += dist * dist;
  }
  return ret > 0 ? sqrt(ret) : 0;
  //return ret > 0 ? ret : 0;
}
