#include <cstdio>
#include <Eigen/Dense>
#include "util/Math.hpp"
#include "util/Assertions.hpp"
#include "../../include/physics/ImplicitODESolver.hpp"

int main() {
    double delta = 1e-4;
    printf("Testing SVD\n");
    Eigen::Matrix2d m;
    m << -0.115855,0.182388,-0.0276305,0.0789723;
    auto r = Math::singularValueDecomposition(m);
    decltype(r) t;
    t << 0.934893,0.35493,0.231022,-0.0177897,-0.511286,-0.859411;
    assertArrayEquals(t, r, delta);
    printf("SVD passed\n");

    double a=-5, fa=16, dfa=-8,
           b=6, fb=49, dfb=14;
    double min = -1;

    auto interpolate = ImplicitODESolver::interpolate;
    assertEquals(min, interpolate(a, fa, dfa, b, fb, dfb), delta);
    assertNotEquals(min + 3, interpolate(a, fa, dfa, b, fb, dfb), delta);
    assertEquals(min, interpolate(b, fb, dfb, a, fa, dfa),delta);
    assertNotEquals(min + 3, interpolate(b, fb, dfb, a, fa, dfa), delta);

    a=1.64852; fa=6.03818; dfa=5.59878;
           b=0.48813; fb=2.17631; dfb=1.66492;
    min = std::nan("");
    assertEquals(min, interpolate(b, fb, dfb, a, fa, dfa),delta);
    assertNotEquals(3, interpolate(b, fb, dfb, a, fa, dfa), delta);

    assertEquals(min, interpolate(a, fa, dfa, b, fb, dfb), delta);
    assertNotEquals(3, interpolate(a, fa, dfa, b, fb, dfb), delta);

    a=0.080544; fa=-0.94141; dfa=0.300789;
           b=1.39375; fb=-0.88757; dfb=0.943421;
    min = 0.987711;
    assertEquals(min, interpolate(b, fb, dfb, a, fa, dfa),delta);
    assertNotEquals(3, interpolate(b, fb, dfb, a, fa, dfa), delta);

    assertEquals(min, interpolate(a, fa, dfa, b, fb, dfb), delta);
    assertNotEquals(3, interpolate(a, fa, dfa, b, fb, dfb), delta);
}
