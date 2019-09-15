#include "physics/FluidDynamics.hpp"
#include "physics/ImplicitODESolver.hpp"
#include "unsupported/Eigen/IterativeSolvers"
#include <random>
#include <iostream>
#include <fstream>
static constexpr int DTEST = 1;
std::mt19937 gen{0};

void rotateLeft(Eigen::Ref<Eigen::VectorXd> v) {
  for(int i=0; i<v.size() - 1; i++) {
    std::swap(v[i], v[i+1]);
  }
}

void rotateRight(Eigen::Ref<Eigen::VectorXd> v) {
  for(int i=v.size()-2; i>=0; i--) {
    std::swap(v[i], v[i+1]);
  }
}

void conjugateGradientSolve(
    const Eigen::VectorXd& rhs, const Eigen::VectorXd& initialGuess,
    std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&)> computeLhs,
    Eigen::VectorXd& result) {
  Eigen::VectorXd temp1(initialGuess.size());
  computeLhs(initialGuess, temp1);  // temp1=Ax0
  Eigen::VectorXd r = rhs - temp1;                  // temp2 = p0 = r0 = b - Ax0
  Eigen::VectorXd p = r;
  double rr1, rr2, alpha, beta;
  computeLhs(p, temp1);  // temp1 = Ap
  rr1 = r.dot(r);//(r * r).sum();
  if(rr1 < 1e-5) {
    result = initialGuess;
    return;
  } 
  alpha = rr1 / p.dot(temp1);
  result = initialGuess + p * alpha;  // result = x1 = x0 + alpha p0

  DMESSAGE("Conjugate Gradient Start.\n");
  for (int k = 0; k < rhs.size(); k++) {
    r = r - alpha * temp1;
    rr2 = r.dot(r);//(r * r).sum();
    // DERROR("Conjugate Gradient: k=%d, ||r||=%g\n", k, rr2);
    if (rr2 < 1e-5) {
      DERROR("Conjugate Gradient: k=%d, ||r||=%g\n", k, rr2);
      DMESSAGE("Conjugate gradient stop.\n");
      break;
    }
    beta = rr2 / rr1;
    rr1 = rr2;
    p = r + beta * p;
    computeLhs(p, temp1);
    alpha = rr1 / p.dot(temp1);//((p * temp1).sum());
    result = result + p * alpha;
  }
}

void FluidDynamicSolver::removeVortex(unsigned i) {
  velocities.erase(velocities.begin() + i);
  vortices.erase(vortices.begin() + 2*i);
  vortices.erase(vortices.begin() + 2*i);
  uVortices.erase(uVortices.begin() + 2*i);
  uVortices.erase(uVortices.begin() + 2*i);
  uVorticesOld.erase(uVorticesOld.begin() + 2*i);
  uVorticesOld.erase(uVorticesOld.begin() + 2*i);
}

void FluidDynamicSolver::emitVortex(double x, double y, double dx, double dy, double omega) {
  vortices.push_back(x);
  vortices.push_back(y);
  velocities.push_back(omega);
  uVortices.push_back(dx);
  uVortices.push_back(dy);
  uVorticesOld.push_back(dx);
  uVorticesOld.push_back(dy);
}
void FluidDynamicSolver::clear() {
  for(int i=0; i<(int)velocities.size(); i++) {
    velocities[i] -= dt*velocities[i]*kDecay;
    if(abs(velocities[i]) < 0.001) {
      removeVortex(i);
      i = i - 1;
    }
  }
}


inline double vectorLength(double x, double y) { return sqrt(x*x + y*y); }

inline void computePanelVelocities(double px0, double py0, double px1,
                                   double py1, double px2, double py2,
                                   double& u, double& v) {

  static const double eps = 0.01;
  Eigen::Vector2d p0(px0, py0), p1(px1, py1), p2(px2, py2);
  Eigen::Vector2d p21 = (p2 - p1);
  Eigen::Vector2d p10 = (p1 - p0);
  Eigen::Vector2d p20 = (p2 - p0);
  double r10 = p10.norm();
  double r20 = p20.norm();
  double r21 = p21.norm();
  double c = (p2-p0).dot(p1-p0)/r20/r10;
  double uR;
  if(c > 1) {
    uR = 0;
  } else if(c < -1) {
    uR = M_PI;
  } else {
    uR = acos(c);
  }
  double kx=px2-px1;
  double ky=py2-py1;
  double theta = atan2(kx, ky) + M_PI_2;
  double vR = log((r20+eps)/(r10+eps));
  // double vR = 0;
  // double nX = (py2 - py1)/r21;
  // double nY = -(px2 - px1)/r21;
  // double hX = nY;
  // double hY = -nX;
  // Eigen::Matrix2d T;
  // auto M = Eigen::Rotation2D(-atan2((p2-p1).x(), (p2-p1).y()));
  Eigen::Vector2d R(uR, vR);
  // T << nX, nY, hX, hY;
  Eigen::Vector2d U = Eigen::Rotation2D(-theta)*R;
  u += U.x();
  v += U.y();
}

inline void getPanelCoordinates(const std::vector<std::vector<const double*>> panels, unsigned i,
                                     double& p0x, double& p0y, double& p1x, double& p1y) {
    unsigned j = i + 1;
    assert(j <= panels[0].size());
    if(j == panels[0].size()) {
       j = 0; 
    }
    p0x = *(panels[0][i] + 0);
    p0y = *(panels[0][i] + 1);
    p1x = *(panels[0][j] + 0);
    p1y = *(panels[0][j] + 1);
}

// this function should be(1 - Exp[-x])/x, but the one below is numerically
// stable
inline double q(double x) { return 1 / (1 + x); }
// sums up all the pparticle velocities at point x,y
inline static void computeParticleVelocities(double x, double y, double sigma,
    std::vector<double>& particles, std::vector<double>& velocities, 
    double& u, double& v) {
    for (unsigned j = 0; j < particles.size() / 2; j++) {
      double xj = particles[2 * j + 0], yj = particles[2 * j + 1];
      double abs2 = (x - xj) * (x - xj) + (y - yj) * (y - yj);
      double c = velocities[j] * q(abs2 / sigma);
      u += (y - yj) * c;
      v += -(x - xj) * c;
    }
}

void FluidDynamicSolver::convect() {
  std::swap(uVortices, uVorticesOld);
  for (unsigned i = 0; i < uVortices.size() / 2; i++) {
    uVortices[2 * i + 0] = uInfinity;
    uVortices[2 * i + 1] = vInfinity;
    double xi = vortices[2 * i + 0], yi = vortices[2 * i + 1];
    computeParticleVelocities(xi, yi, sigma, vortices, velocities, uVortices[2*i+0], uVortices[2*i+1]);
    for(unsigned j=0; j< boundaries[0].size()/DTEST; j ++) {
      double px0, py0, px1, py1;
      getPanelCoordinates(boundaries, j, px0, py0, px1, py1);
      double uP=0, vP=0;
      computePanelVelocities(xi, yi, px0, py0, px1, py1, uP, vP);
      uVortices[2*i+0] += uP*G[j];
      uVortices[2*i+1] += vP*G[j];
    }
  }

  // apply time evolution
  for (unsigned i = 0; i < vortices.size() / 2; i++) {
    vortices[2 * i + 0] += dt * (3. / 2. * uVortices[2 * i + 0] -
                                 1. / 4. * uVorticesOld[2 * i + 0]);
    vortices[2 * i + 1] += dt * (3. / 2. * uVortices[2 * i + 1] -
                                 1. / 2. * uVorticesOld[2 * i + 1]);
  }
}
void FluidDynamicSolver::stretch() {}
void FluidDynamicSolver::diffuse() {
  std::normal_distribution<> d{0, dt * kDiff};
  for (unsigned i = 0; i < vortices.size() / 2; i++) {
    vortices[2 * i + 0] += d(gen);
    vortices[2 * i + 1] += d(gen);
  }
}
void FluidDynamicSolver::addForces() {}
void FluidDynamicSolver::precompute() {
  /* TODO:
  1. Compute the normal components of fluid velocity at the midpoints of
    each panel b_i
  2. Compute the matrix representing the normal components of the
     velocity field induced by panel i at panel j U_ij
  3. Solve the linear equation U_ij g_j = b_j for the panel strengths g_j
  4. Add pannel velocity fields with factors g_j
  */
  unsigned nPanels = boundaries[0].size();
  // Eigen::MatrixXd M(nPanels, nPanels);
  // https://math.stackexchange.com/questions/724697/linear-least-square-optimization-with-linear-equality-constraints
  // Eigen::VectorXd B(nPanels);
  // TODO: optimize because A is very likely antisymmetric
  std::ofstream f;
  f.open("test.dat", std::ofstream::trunc);
  A = Eigen::MatrixXd(nPanels, nPanels);
  // A = Eigen::MatrixXd(nPanels+1, nPanels+1);
  for (unsigned i = 0; i < nPanels; i++) {
    double p0xi, p0yi, p1xi, p1yi;
    getPanelCoordinates(boundaries, i, p0xi, p0yi, p1xi, p1yi);
    double r = vectorLength(p1xi - p0xi, p1yi-p0yi);
    double pNxi = +(p1yi - p0yi)/r;
    double pNyi = -(p1xi - p0xi)/r;
    double pCxi = (p1xi + p0xi)/2;
    double pCyi = (p1yi + p0yi)/2;
    // f << 
    //      pCxi << ' ' << pCyi << ' ' <<
    //      pNxi << ' ' << pNyi << std::endl;
    
    assert(r>1e-9);
    for (unsigned j = 0; j < nPanels; j++) {
      /* compute the normal component of velocity induced by panel j
       at panel i */
      double p0xj, p0yj, p1xj, p1yj;
      getPanelCoordinates(boundaries, j, p0xj, p0yj, p1xj, p1yj);
      double uij=0, vij=0;
      computePanelVelocities(pCxi, pCyi, p0xj, p0yj, p1xj, p1yj, uij, vij);
      A(i, j) = (pNxi * uij + pNyi * vij);
    }
  }
  // precond = A.ldlt().matrixL();
  // precond = precond*precond.transpose();
  // A = A;
  // A(nPanels, nPanels) = 0;
  // const auto M = A.topLeftCorner(nPanels,nPanels);
  // A.topLeftCorner(nPanels, nPanels) = M.transpose()*M;
  // A.topRightCorner(nPanels, 1).fill(1.0);
  // A.bottomLeftCorner(1, nPanels).fill(1.0);
  // std::cout << A << std::endl;
  f << A << std::endl;
  f.close();
}

void FluidDynamicSolver::applyNoSlipBoundaryConditions() {
  for (unsigned i = 0; i < boundaries[0].size(); i++) {
    double p0xi, p0yi, p1xi, p1yi;
    getPanelCoordinates(boundaries, i, p0xi, p0yi, p1xi, p1yi);
    double r = vectorLength(p1xi - p0xi, p1yi-p0yi);
    assert(r>1e-9);
    double pNxi = (p1yi - p0yi)/r;
    double pNyi = -(p1xi - p0xi)/r;
    double pCxi = (p1xi + p0xi)/2;
    double pCyi = (p1yi + p0yi)/2;
    double u=uInfinity, v=vInfinity;
    // computeParticleVelocities(pCxi, pCyi, sigma, vortices, velocities, u, v);
    // double omega = -tau*r;
    double omega = -tau*r*G[i];
    emitVortex(pCxi+pNxi*deltaSlip, pCyi+pNyi*deltaSlip, 0.0, 0.0, omega);
  }
}


void FluidDynamicSolver::applyNoThroughBoundaryConditions() {
  /* TODO:
  1. Compute the normal components of fluid velocity at the midpoints of
    each panel b_i
  2. Compute the matrix representing the normal components of the
     velocity field induced by panel i at panel j U_ij
  3. Solve the linear equation U_ij g_j = b_j for the panel strengths g_j
  4. Add pannel velocity fields with factors g_j
  */
  unsigned nPanels = boundaries[0].size();
  // Eigen::VectorXd B(nPanels + 1);
  Eigen::VectorXd B(nPanels);
  for (unsigned i = 0; i < nPanels; i++) {
    double p0xi, p0yi, p1xi, p1yi;
    getPanelCoordinates(boundaries, i, p0xi, p0yi, p1xi, p1yi);
    double r = vectorLength(p1xi - p0xi, p1yi-p0yi);
    double pNxi = +(p1yi - p0yi)/r;
    double pNyi = -(p1xi - p0xi)/r;
    double pCxi = (p1xi + p0xi)/2;
    double pCyi = (p1yi + p0yi)/2;
    assert(r>1e-9);
    double u=uInfinity, v=vInfinity;
    computeParticleVelocities(pCxi, pCyi, sigma, vortices, velocities, u, v);
    B[i] = -(pNxi * u + pNyi * v);
  }
  // Eigen::HouseholderQR<Eigen::MatrixXd> dec(A);
  Eigen::GMRES<Eigen::MatrixXd> dec(A);
  if(iStep==0 || true) {
    // B.head(nPanels) = A.topLeftCorner(nPanels,nPanels).transpose()*B.head(nPanels);
    // for(int k=0; k<nPanels/4; k++) rotateLeft(bl);
    // bl = A.topLeftCorner(nPanels, nPanels).transpose()*bl;
    // B[nPanels] = 0;
    Eigen::VectorXd initial = 5*Eigen::VectorXd::Random(B.size());//0.11*B;
    // initial[nPanels] = 0;
    // B = precond*B;
    G = dec.solveWithGuess(B, initial);
    // G = A.completeOrthogonalDecomposition().pseudoInverse()*B;
    // G = dec.solve(B);
    std::cout << "Initial "  << A*initial - B << std::endl << std::endl;
    std::cout << "Solution"  << A*G       - B << std::endl << std::endl;
    // G.head(nPanels) = A.topLeftCorner(nPanels,nPanels).transpose()*G.head(nPanels);
    //   conjugateGradientSolve(
    //       B, initial, [&](const auto &lhs, auto &rhs) { rhs = A * lhs; }, G);
  } else {
    // G = dec.solveWithGuess(B, G);
  }
  std::ofstream f;
  f.open("test3.dat");
  f << B;
  f << G;
  f.close();
}

void FluidDynamicSolver::step() {
  assert(vortices.size() == uVortices.size() &&
         uVortices.size() == uVorticesOld.size() &&
         vortices.size() / 2 == velocities.size());
  clear();
  convect();
  diffuse();
  applyNoSlipBoundaryConditions();
  applyNoThroughBoundaryConditions();
  iStep ++;
}

void FluidDynamicSolver::computeFlow(const Eigen::ArrayXXd& X,
                                     const Eigen::ArrayXXd& Y,
                                     Eigen::ArrayXXd& U, Eigen::ArrayXXd& V) {
  unsigned nx = X.rows(), ny = X.cols();
  for (unsigned i = 0; i < nx; i++) {
    for (unsigned j = 0; j < ny; j++) {
      double xi = X(i, j);
      double yi = Y(i, j);
      U(i, j) = uInfinity;
      V(i, j) = vInfinity;
      computeParticleVelocities(xi, yi, sigma, vortices, velocities, U(i, j), V(i, j));
      for(unsigned k=0; k<boundaries[0].size()/DTEST; k++) {
        double px0, py0, px1, py1;
        getPanelCoordinates(boundaries, k, px0, py0, px1, py1);
        double vP=0, uP=0;
        computePanelVelocities(xi, yi, px0, py0, px1, py1, uP, vP);
        U(i,j) += uP*G[k];
        V(i,j) += vP*G[k];
      }
    }
  }
  std::ofstream f;
  f.open("test2.dat", std::ofstream::trunc);
  for (unsigned i = 0; i < boundaries[0].size(); i++) {
    double p0xi, p0yi, p1xi, p1yi;
    getPanelCoordinates(boundaries, i, p0xi, p0yi, p1xi, p1yi);
    double r = vectorLength(p1xi - p0xi, p1yi-p0yi);
    double pNxi = +(p1yi - p0yi)/r;
    double pNyi = -(p1xi - p0xi)/r;
    double pCxi = (p1xi + p0xi)/2;
    double pCyi = (p1yi + p0yi)/2;
    
    double uij=uInfinity, vij=vInfinity;
    assert(r>1e-9);
    computeParticleVelocities(pCxi, pCyi, sigma, vortices, velocities, uij, vij);
    for (unsigned j = 0; j < boundaries[0].size(); j++) {
      /* compute the normal component of velocity induced by panel j
       at panel i */
      double p0xj, p0yj, p1xj, p1yj;
      getPanelCoordinates(boundaries, j, p0xj, p0yj, p1xj, p1yj);
      double uP=0, vP=0;
      computePanelVelocities(pCxi, pCyi, p0xj, p0yj, p1xj, p1yj, uP, vP);
      uij += G[j] * uP;
      vij += G[j] * vP;
    }
    f << 
          pCxi << ' ' << pCyi << ' ' <<
          uij << ' ' <<  vij << ' ' <<
          pNxi << ' ' << pNyi << std::endl;
  }
  f.close();
}