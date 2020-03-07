#pragma once
#include <Eigen/Dense>
#include <cmath>
#include <algorithm>
#include <vector>
#include <numeric>

typedef std::vector<double> VectorD;

static inline unsigned circ(int i, unsigned N)
{
  return (i + N) % N;
}
namespace Math {
static inline VectorD LinSpaced(unsigned N, double a, double b) {
  VectorD lin(N);
  double s = (b-a)/double(N);
  std::generate(lin.begin(), lin.end(), 
    [n=0, s, a, b]() mutable {n++; return a + n*s;}
  );
  return lin;
}

template <typename T>
inline constexpr int signum(T x, std::false_type is_signed) {
  return T(0) < x;
}

template <typename T>
inline constexpr int signum(T x, std::true_type is_signed) {
  return (T(0) < x) - (x < T(0));
}

template <typename T>
inline constexpr int signum(T x) {
  return signum(x, std::is_signed<T>());
}

static bool invertTranspose(const Eigen::Matrix2d& a, Eigen::Matrix2d& inv) {
  double det = a.determinant();
  if (std::isfinite(det)) {
    inv(0, 0) = a(1, 1) / det;
    inv(1, 0) = -a(0, 1) / det;
    inv(0, 1) = -a(1, 0) / det;
    inv(1, 1) = a(0, 0) / det;
    return true;
  }
  return false;
}

static bool invert(const Eigen::Matrix2d& a, Eigen::Matrix2d& inv) {
  double det = a.determinant();
  if (std::isfinite(det)) {
    inv(0, 0) = a(1, 1) / det;
    inv(1, 0) = -a(1, 0) / det;
    inv(0, 1) = -a(0, 1) / det;
    inv(1, 1) = a(0, 0) / det;
    return true;
  }
  return false;
}

static Eigen::Matrix<double, 6, 1> singularValueDecomposition(
    const Eigen::Matrix2d& m) {
  double u1, u2, x, y, v1, v2, e, f, g, h, Q, R, a1, a2, th, phi;
  e = (m(0, 0) + m(1, 1)) / 2;
  f = (m(0, 0) - m(1, 1)) / 2;
  g = (m(1, 0) + m(0, 1)) / 2;
  h = (m(1, 0) - m(0, 1)) / 2;
  Q = hypot(e, h);
  R = hypot(f, g);
  x = Q + R;
  y = Q - R;
  a1 = atan2(g, f);
  a2 = atan2(h, e);
  th = (a2 - a1) / 2;
  phi = (a2 + a1) / 2;
  u1 = cos(phi);
  u2 = sin(phi);
  v1 = cos(th);
  v2 = sin(th);
  Eigen::Matrix<double, 6, 1> result;
  result << u1, u2, x, y, v1, v2;
  return result;
}

static double standardDeviation(const Eigen::ArrayXd& a) {
  int n = a.size();
  double sum = 0;
  double sq_sum = 0;
  for (int i = 0; i < n; ++i) {
    sum += a[i];
  }
  double mean = sum / n;
  for (int i = 0; i < n; ++i) {
    sq_sum += (a[i] - mean) * (a[i] - mean);
  }
  double variance = sq_sum / n;
  return sqrt(variance);
}

static inline double harmonicMean(double a, double b) {
  return 2 * a * b / (a + b + 1e-9);
}
static inline double sqrd(double a) { return a * a; }

};  // namespace Math


static inline VectorD& operator*=(VectorD& v1, const VectorD& v2) {
  assert(v1.size() == v2.size());
  std::transform(v1.begin(), v1.end(), 
                 v2.begin(), v1.begin(),
                 std::multiplies<double>() );
  return v1;
}

static inline VectorD& operator+=(VectorD& v1, const VectorD& v2) {
  assert(v1.size() == v2.size());
  std::transform(v1.begin(), v1.end(), 
                 v2.begin(), v1.begin(),
                 std::plus<double>() );
  return v1;
}

static inline VectorD& operator-=(VectorD& v1, const VectorD& v2) {
  assert(v1.size() == v2.size());
  std::transform(v1.begin(), v1.end(), 
                 v2.begin(), v1.begin(),
                 std::minus<double>() );
  return v1;
}

static inline double sum(const VectorD& v1) {
  return std::accumulate(v1.begin(), v1.end(), 0.0);
}

static inline void negate(VectorD& v1) {
  std::transform(v1.begin(), v1.end(), 
                 v1.begin(), std::negate<double>());
}

static inline VectorD operator-(const VectorD& v1, const VectorD& v2) {
  assert(v1.size() == v2.size());
  VectorD tmp(v1.size());
  std::transform(v1.begin(), v1.end(), 
                 v2.begin(), tmp.begin(),
                 std::minus<double>() );
  return tmp;
}


static inline VectorD operator*(const VectorD& v1, const VectorD& v2) {
  assert(v1.size() == v2.size());
  VectorD tmp(v1.size());
  std::transform(v1.begin(), v1.end(), 
                 v2.begin(), tmp.begin(),
                 std::multiplies<double>() );
  return tmp;
}

static inline VectorD operator+(const VectorD& v1, const VectorD& v2) {
  assert(v1.size() == v2.size());
  VectorD tmp(v1.size());
  std::transform(v1.begin(), v1.end(), 
                 v2.begin(), tmp.begin(),
                 std::plus<double>() );
  return tmp;
}

static inline VectorD operator+(const VectorD& v1, double x) {
  VectorD tmp(v1.size());
  std::transform(v1.begin(), v1.end(), 
                 tmp.begin(),
                 [&](double y){return y+x;} );
  return tmp;
}

static inline VectorD operator*(const VectorD& v1, double x) {
  VectorD tmp(v1.size());
  std::transform(v1.begin(), v1.end(), 
                 tmp.begin(), [&](double a){return a*x;} );
  return tmp;
}

static void setZero(VectorD& v1) {
  std::fill(v1.begin(), v1.end(), 0.0);
}


