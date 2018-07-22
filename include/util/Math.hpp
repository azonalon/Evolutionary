#pragma once
#include <cmath>
#include <Eigen/Dense>

namespace Math {
    template <typename T> inline constexpr
    int signum(T x, std::false_type is_signed) {
        return T(0) < x;
    }

    template <typename T> inline constexpr
    int signum(T x, std::true_type is_signed) {
        return (T(0) < x) - (x < T(0));
    }

    template <typename T> inline constexpr
    int signum(T x) {
        return signum(x, std::is_signed<T>());
    }
    // static std::string str(Eigen::Matrix2d m) {
    //     return std::string.format("((%g, %g), (%g, %g))", m(0,0), m(0,1), m(1,0), m(1,1));
    // }
    // static void add(double a, Eigen::Matrix2d A, double b, Eigen::Matrix2d B,
    //     Eigen::Matrix2d dest) {
    //         dest(0,0) = a * A(0,0) + b * B(0,0);
    //         dest(0,1) = a * A(0,1) + b * B(0,1);
    //         dest(1,0) = a * A(1,0) + b * B(1,0);
    //         dest(1,1) = a * A(1,1) + b * B(1,1);
    //     }
        static bool invertTranspose(const Eigen::Matrix2d& a , Eigen::Matrix2d& inv ) {

            double det = a.determinant();
            if(std::isfinite(det)) {
                inv(0,0) = a(1,1)/det;
                inv(1,0) =-a(0,1)/det;
                inv(0,1) =-a(1,0)/det;
                inv(1,1) = a(0,0)/det;
                return true;
            }
            return false;
        }

        static bool invert(const Eigen::Matrix2d& a, Eigen::Matrix2d& inv) {

            double det = a.determinant();
            if(std::isfinite(det)) {
                inv(0,0) = a(1,1)/det;
                inv(1,0) =-a(1,0)/det;
                inv(0,1) =-a(0,1)/det;
                inv(1,1) = a(0,0)/det;
                return true;
            }
            return false;
        }

        static Eigen::Matrix<double,6,1> singularValueDecomposition(const Eigen::Matrix2d& m) {
            double u1, u2, x, y, v1, v2, e, f, g, h, Q, R, a1, a2, th, phi;
            e = (m(0,0) + m(1,1))/2;
            f = (m(0,0) - m(1,1))/2;
            g = (m(1,0) + m(0,1))/2;
            h = (m(1,0) - m(0,1))/2;
            Q = hypot(e, h);
            R = hypot(f, g);
            x = Q + R;
            y = Q - R;
            a1 = atan2(g, f);
            a2 = atan2(h, e);
            th = (a2 - a1)/2;
            phi = (a2 + a1)/2;
            u1 = cos(phi);
            u2 = sin(phi);
            v1 = cos(th);
            v2 = sin(th);
            Eigen::Matrix<double,6,1> result;
            result << u1, u2, x, y, v1, v2;
            return result;
        }

        static double standardDeviation(const Eigen::ArrayXd& a) {
            int n = a.size();
            double sum = 0;
            double sq_sum = 0;
            for(int i = 0; i < n; ++i) {
                sum += a[i];
            }
            double mean = sum / n;
            for(int i = 0; i < n; ++i) {
                sq_sum += (a[i]-mean) * (a[i]-mean);
            }
            double variance = sq_sum / n;
            return sqrt(variance);
        }
        // double mod(double x, double n) {
        //     return (x % n) - (x < 0 ? n : 0);
        // }
        // static double circleMod(double phi) {
        //     if(phi < -PI) {
        //         return phi + 2 * PI;
        //     } else if(phi > PI) {
        //         return phi - 2 * PI;
        //     }
        //     return phi;
        // }

        // static double average(Eigen::ArrayXd a) {
        //     int n = a.size;
        //     double sum = 0;
        //     for(int i = 0; i < n; ++i) {
        //         sum += a[i];
        //     }
        //     return sum / n;
        // }

        // static double[] getColumn(double[][] a, int c) {
        //     // int m = a[0].length;
        //     double[] col = new double[a.length];
        //     for(int i = 0; i < a.length; i++) {
        //         col[i] = a[i][c];
        //     }
        //     return col;
        // }

        static double harmonicMean(double a, double b) {
            return 2 * a * b / (a + b + 1e-9);
        }
        static double sqrd(double a) {
            return a*a;
        }
    }
