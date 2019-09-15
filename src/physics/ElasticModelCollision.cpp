#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graphviz.hpp>
#include <exception>
#include <fstream>
#include "physics/ElasticModel.hpp"

using namespace Eigen;
static double c = 1000;

double squarePointLineSegmentDistance(double px, double py, double p0x,
                                      double p0y, double p1x, double p1y) {
  double dX10 = p1x - p0x;
  double dY10 = p1y - p0y;
  double dX0 = px - p0x;
  double dY0 = py - p0y;
  double s = (dX10 * dY0 - dX0 * dY10);
  double Lsquare = dX10 * dX10 + dY10 * dY10;
  return s * s / Lsquare;
};

bool pointInTriangle(double px, double py, double p0x, double p0y, double p1x,
                     double p1y, double p2x, double p2y) {
  double dX2 = px - p2x;
  double dY2 = py - p2y;
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
                                       const Eigen::ArrayXd& x,
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

  double s = -dX10 * dY0 + dX0 * dY10;
  double s2 = s * s;

  double L2 = dX10 * dX10 + dY10 * dY10;
  double L4 = L2 * L2;
  double L6 = L4 * L2;

  double eps = 1e-15;
  double U2 = s2 / L2 + eps;
  double U = sqrt(U2);
  double phi = s2 / L2 * U;

  // double q1 = s2 + 2*L2 * U2;
  // double q2 = q1 + 6*s2;
  // double q3 = 6*q1*s + 4 *s2*s;

  double kf = s2 * s / (L6 * U) + 2 * s * U / L4;

  dest[iPoint + 0] += c * kf * (dY10 * L2);
  dest[iPoint + 1] += c * kf * (-dX10 * L2);
  dest[i + 0] += c * kf * (dY1 * L2 + dX10 * s);
  dest[i + 1] += c * kf * (-dX1 * L2 + dY10 * s);
  dest[j + 0] += c * kf * (-dY0 * L2 - dX10 * s);
  dest[j + 1] += c * kf * (dX0 * L2 - dY10 * s);

  return c * phi;
}

void addSelfCollisionPenaltyGradientDifferential(std::array<unsigned, 3> triplet,
                                              const Eigen::ArrayXd& x,
                                              const Eigen::ArrayXd& dx,
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
  dest(iPoint + 0) += c * df(0);
  dest(iPoint + 1) += c * df(1);
  dest(i + 0) += c * df(2);
  dest(i + 1) += c * df(3);
  dest(j + 0) += c * df(4);
  dest(j + 1) += c * df(5);
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
    E += addSelfCollisionPenaltyGradient(triplet, x, dest);
  }
  return E;
};

// This function computes the list containing collisions to be resolved during
// the energy function optimization step
void ElasticModel::populateSelfCollisionList(const Eigen::ArrayXd& x) {
  selfCollisionList.clear();
  for (unsigned l = 0; l < m; l++) {
    unsigned int i = 2 * Te(l, 0), j = 2 * Te(l, 1), k = 2 * Te(l, 2);
    for (unsigned n = 0; n < m; n++) {
      if (n == l) {
        continue;
      }
      for (unsigned jj = 0; jj < 3; jj++) {
        unsigned int iPoint = 2 * Te(n, jj);
        if (iPoint != i && iPoint != j && iPoint != k &&
            pointInTriangle(x(iPoint + 0), x(iPoint + 1), x(i + 0), x(i + 1),
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
          if (d1 < d2 && d1 < d3) {
            selfCollisionList.push_back({iPoint, i, j});
          } else if (d2 < d3 && d2 < d1) {
            selfCollisionList.push_back({iPoint, i, k});
          } else {
            selfCollisionList.push_back({iPoint, j, k});
          }
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
    addSelfCollisionPenaltyGradientDifferential(triplet, x, dx, dest);
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
  // typedef std::pair<unsigned, unsigned> Edge;
  // std::multimap<unsigned, unsigned> adjacentTriangles;
  for (unsigned l = 0; l < m; l++) {
    std::vector<bool> isAdjacent(3, false);
    // std::array<Edge, 3> edges {
    //     Edge{Te(l, 0), Te(l, 1)},
    //     Edge{Te(l, 1), Te(l, 2)},
    //     Edge{Te(l, 2), Te(l, 0)}
    // };
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
  std::ofstream of1("test1.dot");
  boost::write_graphviz(of1, g);
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
}
