#include <Eigen/Dense>
#include <vector>

class FluidDynamicSolver {
 public:
  std::vector<double> vortices;
  std::vector<double> velocities;
  std::vector<double> uVortices;
  std::vector<double> uVorticesOld;
  long nX, nY;
  double dt = 0.01;
  double kDiff = 1;
  double sigma = 0.1;
  double tau = 0;
  double kDecay = 100;
  double deltaSlip = 0.1;
  unsigned iStep=0;
  double uInfinity=0, vInfinity=0;
  Eigen::MatrixXd precond;
  const std::vector<std::vector<const double*>>& boundaries;
  Eigen::VectorXd G;
  Eigen::MatrixXd A;
  void clear();
  void convect();
  void stretch();
  void diffuse();
  void addForces();
  void precompute();
  void emitVortex(double x, double y, double dx, double dy, double omega);
  void removeVortex(unsigned i);
  void applyNoThroughBoundaryConditions();
  void applyNoSlipBoundaryConditions();
  void step();
  void computeFlow(
    const Eigen::ArrayXXd& X, const Eigen::ArrayXXd& Y,
    Eigen::ArrayXXd& U, Eigen::ArrayXXd& V
  );
  FluidDynamicSolver(const std::vector<std::vector<const double*>>& boundaries)
      : boundaries(boundaries), G(boundaries[0].size()) {
        precompute();
        G.fill(1);
  }
};