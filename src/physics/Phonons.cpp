#include "physics/Phonons.hpp"
#include <stdio.h>
#include <cmath>

void Phonons::update(double t) {
for(unsigned i=0; i<m.rows(); i++) {
    for(unsigned j=0; j<m.cols(); j++) {
    m(i, j) = 1 + 0.2*sin(t + 4*(double)i/(double)m.rows());
    }
}
}