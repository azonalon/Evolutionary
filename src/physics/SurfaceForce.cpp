#include "physics/ElasticModel.hpp"

void ElasticModel::setSurfaceForce(unsigned iObject, unsigned iSurface, double force) {
  //TODO handle last surface vertex and optimize redundancy for swapping j and i
  assert(iObject < surfaces.size());
  assert(iSurface < surfaces[iObject].size());

  unsigned int i,j;
  i = surfaces[iObject][iSurface];
  if(iSurface == surfaces[iObject].size() - 1) {
    j = surfaces[iObject][0];
  } else{
    j = surfaces[iObject][iSurface+1];
  }
  auto v =  Eigen::Vector2d(x0[2 * i + 0] - x0[2 * j + 0], x0[2 * i + 1] - x0[2 * j + 1]);
  double px = v[0];
  double py = v[1];
  fExt[2 * i + 0]  =  force * ( - py );
  fExt[2 * i + 1]  =  force * ( + px );
  fExt[2 * j + 0]  =  force * ( - py );
  fExt[2 * j + 1]  =  force * ( + px );
}