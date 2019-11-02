#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graphviz.hpp>
#include <exception>
#include <fstream>
#include "physics/ElasticModel.hpp"

using namespace Eigen;
double strengthSelfCollision = 100000;
double muSelfFriction = 1000;
#ifndef SELF_COLLISION_FRICTION_ENABLE
#define SELF_COLLISION_FRICTION_ENABLE true
#endif

static inline double clampOne(double x) {
  if(x <= 1) {
    return x;
  }
  return 1;
}

static inline double squarePointPointDistance(double px, double py, double p0x,
                                              double p0y) {
  return ((px - p0x) * (px - p0x) + (py - p0y) * (py - p0y));
}
static inline double squarePointLineSegmentDistance(double px, double py, double p0x,
                                      double p0y, double p1x, double p1y) {
  double dX10 = p1x - p0x;
  double dY10 = p1y - p0y;
  double dX0 = px - p0x;
  double dY0 = py - p0y;
  double s = (dX10 * dY0 - dX0 * dY10);
  double Lsquare = dX10 * dX10 + dY10 * dY10;
  return s * s / Lsquare;
};

static inline bool pointInTriangle(double px, double py, double p0x, double p0y, double p1x,
                     double p1y, double p2x, double p2y) {
  double dX2  = px  - p2x;
  double dY2  = py  - p2y;
  double dX12 = p1x - p2x;
  double dY12 = p1y - p2y;
  double dX02 = p0x - p2x;
  double dY02 = p0y - p2y;

  double D = dY12 * dX02 - dX12 * dY02;
  double s = dY12 * dX2 - dX12 * dY2;
  double t = -dY02 * dX2 + dX02 * dY2;
  if (D < 0) {
    return s <= 0 && t <= 0 && s + t >= D;
  }
  return s >= 0 && t >= 0 && s + t <= D;
}

// adds force resulting from point iPoint inside surface (surface is a line from
// vertex i to j)
double addSelfCollisionPenaltyGradient(std::array<unsigned, 3> triplet,
                                       const Eigen::ArrayXd& MI,
                                       const Eigen::ArrayXd& x,
                                       Eigen::ArrayXd& dest,
                                       Eigen::ArrayXd& v) {
  unsigned &iPoint = triplet[0], i = triplet[1], j = triplet[2];
  double px = x(iPoint), py = x(iPoint + 1);
  double p0x = x(i), p0y = x(i + 1);
  double p1x = x(j), p1y = x(j + 1);
  double dX1 = px - p1x;
  double dY1 = py - p1y;
  double dX0 = px - p0x;
  double dY0 = py - p0y;
  double dX10 = p1x - p0x;
  double dY10 = p1y - p0y;



  double s = -dX10 * dY0 + dX0 * dY10;
  double s2 = s * s;

  double L2 = dX10 * dX10 + dY10 * dY10;
  double L1 = sqrt(dX10 * dX10 + dY10 * dY10);
  double L4 = L2 * L2;
  double L6 = L4 * L2;

  // U2 is the square of the point line distance
  double eps = 1e-15;
  double U2  = s2 / L2;
  double U   = sqrt(s2 / L2+eps);
  double phi = strengthSelfCollision*U2*U;

  double kf = strengthSelfCollision *( s2 * s / (L6 * U) + 2 * s * U / L4);

  dest[iPoint + 0] += kf * (dY10 * L2);
  dest[iPoint + 1] += kf * (-dX10 * L2);
  dest[i + 0]      += kf * (dY1 * L2 + dX10 * s);
  dest[i + 1]      += kf * (-dX1 * L2 + dY10 * s);
  dest[j + 0]      += kf * (-dY0 * L2 - dX10 * s);
  dest[j + 1]      += kf * (dX0 * L2 - dY10 * s);

  // apply friction
  #if SELF_COLLISION_FRICTION_ENABLE
  double& v0x = v(iPoint);
  double& v0y = v(iPoint + 1);
  double& v1x = v(i);
  double& v1y = v(i+ 1);
  double& v2x = v(j);
  double& v2y = v(j+ 1);
  const double d02 = dX0*dX0+dY0*dY0;
  const double w0 = -1;
  const double w2 = sqrt((d02 - U2)/L2);
  const double w1 = 1-w2;
  const double mbarInv = w0*w0*MI[iPoint] + w1*w1*MI[i] + w2*w2*MI[j];
  const double vbarx = mbarInv * (w0 * v0x + w1 * v1x + w2 * v2x);
  const double vbary = mbarInv * (w0 * v0y + w1 * v1y + w2 * v2y);
  double jx = dX10*vbarx/L1/mbarInv;
  double jy = dY10*vbary/L1/mbarInv;
  double jL = sqrt(jx*jx + jy*jy);
  double frictionFactor = clampOne(muSelfFriction * abs(kf) / jL);
  v0x  = v0x + w0 * MI[iPoint] * frictionFactor * jx;
  v0y  = v0y + w0 * MI[iPoint] * frictionFactor * jy;
  v1x  = v1x + w1 * MI[i     ] * frictionFactor * jx;
  v1y  = v1y + w1 * MI[i     ] * frictionFactor * jy;
  v2x  = v2x + w2 * MI[j     ] * frictionFactor * jx;
  v2y  = v2y + w2 * MI[j     ] * frictionFactor * jy;
  #endif

  return phi;

}

void addSelfCollisionPenaltyGradientDifferential(std::array<unsigned, 3> triplet,
                                              const Eigen::ArrayXd& MI,
                                              const Eigen::ArrayXd& x,
                                              const Eigen::ArrayXd& dx,
                                              const Eigen::ArrayXd& v,
                                              Eigen::ArrayXd& dest) {
  unsigned &iPoint = triplet[0], i = triplet[1], j = triplet[2];
  double px = x(iPoint), py = x(iPoint + 1);
  double p0x = x(i), p0y = x(i + 1);
  double p1x = x(j), p1y = x(j + 1);
  double dX1 = px - p1x;
  double dY1 = py - p1y;
  double dX0 = px - p0x;
  double dY0 = py - p0y;
  double dX10 = p1x - p0x;
  double dY10 = p1y - p0y;
  double L2 = dX10 * dX10 + dY10 * dY10;
  double L4 = L2 * L2;
  double L6 = L4 * L2;
  double L10 = L4 * L6;
  double s = -dX10 * dY0 + dX0 * dY10;
  double s2 = s * s;
  double eps = 1e-15;
  double U2 = s2 / L2 + eps;
  double U = sqrt(U2);
  double U3 = U2 * U;
  double U4 = U2 * U2;

  double q1 = s2 + 2 * L2 * U2;
  double q2 = q1 + 6 * s2;
  double q3 = 6 * q1 * s + 4 * s2 * s;

  double kf = s2 * s / (L6 * U) + 2 * s * U / L4;

  Eigen::Matrix<double, 6, 1> f;
  Eigen::Matrix<double, 6, 1> kfd;
  Eigen::Matrix<double, 6, 6> fd;

  f << dY10 * L2, -dX10 * L2, dY1 * L2 + dX10 * s, -dX1 * L2 + dY10 * s,
      -dY0 * L2 - dX10 * s, dX0 * L2 - dY10 * s;

  kfd << -(dY10 * L2 * q1 * s2) + dY10 * L4 * q2 * U2,
      dX10 * L2 * q1 * s2 - dX10 * L4 * q2 * U2,
      -(q1 * (dY1 * L2 + dX10 * s) * s2) +
          L2 * (dY1 * L2 * q2 + dX10 * q3) * U2 - 4 * dX10 * L4 * s * U4,
      q1 * (dX1 * L2 - dY10 * s) * s2 +
          L2 * (-(dX1 * L2 * q2) + dY10 * q3) * U2 - 4 * dY10 * L4 * s * U4,
      q1 * (dY0 * L2 + dX10 * s) * s2 - L2 * (dY0 * L2 * q2 + dX10 * q3) * U2 +
          4 * dX10 * L4 * s * U4,
      q1 * (-(dX0 * L2) + dY10 * s) * s2 +
          L2 * (dX0 * L2 * q2 - 2 * dY10 * s * (3 * q1 + 2 * s2)) * U2 +
          4 * dY10 * L4 * s * U4;

  fd << 0, 0, -2 * dX10 * dY10, -dX10 * dX10 - 3 * dY10 * dY10, 2 * dX10 * dY10,
      dX10 * dX10 + 3 * dY10 * dY10, 0, 0, 3 * dX10 * dX10 + dY10 * dY10,
      2 * dX10 * dY10, -3 * dX10 * dX10 - dY10 * dY10, -2 * dX10 * dY10,
      dX10 * dY10, dY10 * dY10, dX10 * dY0 - dX10 * dY1 - dX0 * dY10,
      -(dX1 * dX10) - 2 * dY1 * dY10,
      -2 * dX10 * dY0 + 2 * dX10 * dY1 + dX0 * dY10,
      dX0 * dX10 + 2 * dY1 * dY10 - dX10 * dX10 - dY10 * dY10, -dX10 * dX10,
      -(dX10 * dY10), 2 * dX1 * dX10 + dY1 * dY10,
      dX10 * dY0 - dX0 * dY10 + dX1 * dY10, -2 * dX1 * dX10 - dY0 * dY10 + L2,
      -(dX10 * dY0) + 2 * dX0 * dY10 - 2 * dX1 * dY10, -(dX10 * dY10),
      -dY10 * dY10, dX10 * dY0 - dX10 * dY1 + dX0 * dY10,
      dX1 * dX10 + 2 * dY0 * dY10 + L2, -(dX0 * dY10),
      -(dX0 * dX10) - 2 * dY0 * dY10, dX10 * dX10, dX10 * dY10,
      -2 * dX0 * dX10 - dY1 * dY10 - dX10 * dX10 - dY10 * dY10,
      -(dX10 * dY0) - dX0 * dY10 + dX1 * dY10, 2 * dX0 * dX10 + dY0 * dY10,
      dX10 * dY0;

  Eigen::Matrix<double, 6, 6> h =
      1 / (L10 * U3) * f * kfd.transpose() + kf * fd;
  Eigen::Matrix<double, 6, 1> dxtmp;
  dxtmp << dx(iPoint), dx(iPoint + 1), dx(i), dx(i + 1), dx(j), dx(j + 1);
  Eigen::Matrix<double, 6, 1> df = h * dxtmp;
  dest(iPoint + 0) += strengthSelfCollision * df(0);
  dest(iPoint + 1) += strengthSelfCollision * df(1);
  dest(i + 0)      += strengthSelfCollision * df(2);
  dest(i + 1)      += strengthSelfCollision * df(3);
  dest(j + 0)      += strengthSelfCollision * df(4);
  dest(j + 1)      += strengthSelfCollision * df(5);
}

double ElasticModel::computeCollisionPenaltyGradient(const Eigen::ArrayXd& x,
                                                     Eigen::ArrayXd& dest) {
  double E = 0;
  // first loop: object collision
  for (unsigned i = 0; i < x.size() / 2; i++) {
    double px = x[2 * i + 0], py = x[2 * i + 1];
    for (auto& c : collisionObjects) {
      if (c->boundingBox().contains(px, py)) {
        // TODO: this always uses mass in x direction
        E += c->computePenaltyGradient(&x[2 * i], &v[2 * i], dt / M[2 * i],
                                       &dest[2 * i]);
      }
    }
  }
  for (auto& triplet : selfCollisionList) {
    E += addSelfCollisionPenaltyGradient(triplet, MI, x, dest, v);
  }
  return E;
};

std::array<unsigned, 3>
ElasticModel::closestSurfaceFromPoint(unsigned iPoint, unsigned iSurface,
                                      const Eigen::ArrayXd &x) {
  unsigned n = surfaces[iSurface].size();
  auto& surface = surfaces[iSurface];
  double dMin = 1e33;
  unsigned i0Min=0, i1Min=1;
  for(int i=0; i < n; i++) {
    unsigned i0 = 2*surface[i];
    unsigned i1 = 2*surface[(i+1)%n];
    double px = x[iPoint], py=x[iPoint+1];
    double p0x = x[i0], p0y=x[i0+1];
    double p1x = x[i1], p1y=x[i1+1];
    double d = squarePointLineSegmentDistance(px, py, p0x, p0y, p1x, p1y); 
    if(d<dMin) {
      i0Min=i0;
      i1Min=i1;
      dMin = d;
    }
  }
  return {iPoint, i0Min, i1Min};
}
// This function computes the list containing collisions to be resolved during
// the energy function optimization step
void ElasticModel::populateSelfCollisionList(const Eigen::ArrayXd& x) {
  selfCollisionList.clear();
  for (unsigned l = 0; l < m; l++) {
    unsigned int i = 2 * Te(l, 0), j = 2 * Te(l, 1), k = 2 * Te(l, 2);
    for (unsigned u = 0; u < n/2; u++) {
      unsigned int iPoint = 2 * u;
      if (iPoint == i || iPoint == j || iPoint == k) {
        continue;
      }
      /**
       * Compute the distance to all triangle edges. If the closest or the second
       * closest edge is a surface edge, add it to the collision list. Else look
       * for the closest surface edge withing the given surface.
      */
      if (pointInTriangle(x(iPoint + 0), x(iPoint + 1), x(i + 0), x(i + 1),
                          x(j + 0), x(j + 1), x(k + 0), x(k + 1))) {
        double d1 = squarePointLineSegmentDistance(
            x(iPoint + 0), x(iPoint + 1), x(i + 0), x(i + 1), x(j + 0),
            x(j + 1));
        double d2 = squarePointLineSegmentDistance(
            x(iPoint + 0), x(iPoint + 1), x(i + 0), x(i + 1), x(k + 0),
            x(k + 1));
        double d3 = squarePointLineSegmentDistance(
            x(iPoint + 0), x(iPoint + 1), x(j + 0), x(j + 1), x(k + 0),
            x(k + 1));
        if (d1 < d2 && d1 < d3 && isSurfaceEdge[{i/2,j/2}]) {
            selfCollisionList.push_back({iPoint, i, j});
          }
        else if ((d2 < d3) && isSurfaceEdge[{i/2, k/2}]) {
            selfCollisionList.push_back({iPoint, i, k});
        }  
        else if ((d3 < d2) && isSurfaceEdge[{j/2, k/2}]) {
            selfCollisionList.push_back({iPoint, j, k});
        }  else {
            unsigned iSurface = surfaceFromVertex[i/2];
            selfCollisionList.push_back(closestSurfaceFromPoint(iPoint, iSurface, x));
        }
      }
    }
  }
}


void ElasticModel::computeCollisionPenaltyGradientDifferential(
    const Eigen::ArrayXd& x, const Eigen::ArrayXd& dx, Eigen::ArrayXd& dest) {
  for (unsigned i = 0; i < x.size() / 2; i++) {
    double px = x[2 * i + 0], py = x[2 * i + 1];
    for (CollisionObject* c : collisionObjects) {
      if (c->boundingBox().contains(px, py)) {
        c->computePenaltyGradientDifferential(&x[2 * i], &dx[2 * i],
                                              &dest[2 * i]);
      }
    }
  }
  for (auto& triplet : selfCollisionList) {
    addSelfCollisionPenaltyGradientDifferential(triplet, MI, x, dx, v, dest);
  }
};

// typedef boost::graph_traits<Graph>::edge_iterator edge_iterator;

struct VertexProperties {
  bool visited = false;
};

typedef boost::adjacency_list<boost::listS, boost::vecS, boost::directedS,
                              VertexProperties>
    Graph;


// This function computes the surfaces of the elastic model
void ElasticModel::collisionPrecompute(const std::vector<double>& vertices) {
  Graph g;
  /**
   * Go through all pairs of triangles. If a a triangle has an edge without
   * an adjacent triangle, this edge is a  surface edge. Add all vertices
   * contained by surface edges to the graph g
   */
  for (unsigned l = 0; l < m; l++) {
    std::vector<bool> isAdjacent(3, false);
    for (unsigned u = 0; u < m; u++) {
      if (u == l) continue;
      for (int i = 0; i < 3; i++) {
        if ((Te(l, i) == Te(u, 0) && Te(l, (i + 1) % 3) == Te(u, 1)) ||
            (Te(l, i) == Te(u, 1) && Te(l, (i + 1) % 3) == Te(u, 0)) ||
            (Te(l, i) == Te(u, 1) && Te(l, (i + 1) % 3) == Te(u, 2)) ||
            (Te(l, i) == Te(u, 2) && Te(l, (i + 1) % 3) == Te(u, 1)) ||
            (Te(l, i) == Te(u, 2) && Te(l, (i + 1) % 3) == Te(u, 0)) ||
            (Te(l, i) == Te(u, 0) && Te(l, (i + 1) % 3) == Te(u, 2))) {
          isAdjacent[i] = true;
        }
      }
    }
    /**
     * Adds edges to the graph where on one side of the edge there is no adjacent
     * triangle
     */
    auto addEdge = [&](int i, int j, int k) {
      const double& px0 = vertices[2 * Te(l, i) + 0];
      const double& py0 = vertices[2 * Te(l, i) + 1];
      const double& px1 = vertices[2 * Te(l, j) + 0];
      const double& py1 = vertices[2 * Te(l, j) + 1];
      const double& px2 = vertices[2 * Te(l, k) + 0];
      const double& py2 = vertices[2 * Te(l, k) + 1];
      double s = px2 * (-py0 + py1) + px1 * (py0 - py2) + px0 * (-py1 + py2);
      if (s < 0) {
        boost::add_edge(Te(l, j), Te(l, k), g);
      } else {
        boost::add_edge(Te(l, k), Te(l, j), g);
      }
    };
    if (isAdjacent[0] == false) {
      addEdge(2, 0, 1);
    }
    if (isAdjacent[1] == false) {
      addEdge(0, 1, 2);
    }
    if (isAdjacent[2] == false) {
      addEdge(1, 2, 0);
    }
  }
  /* Iterate over all the previously created edges to make adjacent chains of
   vertices which consist of the distinct surfaces of the model 
   */
  // std::ofstream of1("test1.dot");
  // boost::write_graphviz(of1, g);
  auto ei = boost::vertices(g);
  for (auto it = ei.first; it != ei.second; it++) {
    auto chain = *it;
    if (!g[chain].visited) {
      std::vector<unsigned int> surface;
      do {
        g[chain].visited = true;
        surface.push_back(chain);
        auto adjacent = boost::adjacent_vertices(chain, g);
        chain = *adjacent.first;
      } while (!g[chain].visited);
      if(surface.size() > 1) { 
        surfaces.push_back(surface);
      }
    }
  }
  // populate the isSurfaceEdge list
  for (unsigned u = 0; u < m; u++) {
    using edge = std::array<unsigned, 2>;
    auto triangleEdges = {
      edge{Te(u, 0), Te(u,1)},
      edge{Te(u, 1), Te(u,0)},
      edge{Te(u, 1), Te(u,2)},
      edge{Te(u, 2), Te(u,1)},
      edge{Te(u, 2), Te(u,0)},
      edge{Te(u, 0), Te(u,2)}
    };
    for(auto& e:triangleEdges) {
      isSurfaceEdge[e] = 0; 
    }
  }
  for(auto& s: surfaces) {
    for(unsigned i=0; i< s.size(); i++) {
      isSurfaceEdge[{s[i], s[(i+1)%s.size()]}] = 1; 
      isSurfaceEdge[{s[(i+1)%s.size()], s[i]}] = 1; 
    }
  }
  // for(auto& e: isSurfaceEdge) {
    // printf("Surface Edge %d,%d? -> %d\n", e.first[0], e.first[1], e.second);
  // }
  // Compute the closest surface from edge map. The edge-edge distance is defined
  // as the minimum of the average of the two point distances.
  for (unsigned u = 0; u < m; u++) {
    using edge = std::array<unsigned, 2>;
    auto triangleEdges = {
      edge{Te(u, 0), Te(u,1)},
      edge{Te(u, 1), Te(u,2)},
      edge{Te(u, 2), Te(u,0)}
    };
    for(auto& e: triangleEdges) {
      double shortestEdgeDistance = 1e33;
      double edgeDistance = 1e33;
      Eigen::Vector2d v0(vertices[2*e[0] + 0], vertices[2*e[0] + 1]);
      Eigen::Vector2d v1(vertices[2*e[1] + 0], vertices[2*e[1] + 1]);
      unsigned j=0;
      for(auto& s:  surfaces) {
        for(unsigned i=0; i<s.size(); i++) {
          unsigned n0 = 2*s[i];
          unsigned n1 = 2*(s[(i + 1) % s.size()]);
          Eigen::Vector2d v2(vertices[n0], vertices[n0+1]);
          Eigen::Vector2d v3(vertices[n1], vertices[n1+1]);
          double d02 = (v0-v2).norm();
          double d03 = (v0-v3).norm();
          double d12 = (v1-v2).norm();
          double d13 = (v1-v3).norm();
          double d1 = (d02 + d13)/2.0;
          double d2 = (d03 + d12)/2.0;
          if(d1 < d2) {
            edgeDistance = d1;
          } else  {
            edgeDistance = d2;
          }
          if(edgeDistance < shortestEdgeDistance) {
            shortestEdgeDistance = edgeDistance;
            closestSurfaceFromEdge[e] = {s[i], s[(i+1)%s.size()]};
            surfaceFromVertex[e[0]] = j;
            surfaceFromVertex[e[1]] = j;
          }
        }
      j++;
      }
      auto csf = closestSurfaceFromEdge[e];
      // printf("Edge %d,%d -> %d,%d\n", e[0], e[1], closestSurfaceFromEdge[e][0],
      //                                             closestSurfaceFromEdge[e][1]);
    }
  }
}
