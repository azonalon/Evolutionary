#include "physics/InvertibleNeoHookeanModel.hpp"

static double square(double x) {
    return x*x;
}
static double cube(double x) {
    return x*x*x;
}
static double quart(double x) {
    return x*x*x*x;
}

Eigen::Matrix<double,6,1> InvertibleNeoHookeanModel::computeStressDifferentialComponents(
						double sx, double sy, double l,
						double mu, double eps)
{
	double ux = sx - 1;
	double uy = sy - 1;
	double ux2 = ux*ux;
	double ux3 = ux2*ux;
	double ux4 = ux2*ux2;
	double ux5 = ux3*ux2;
	double uy2 = uy*uy;
	double uy3 = uy2*uy;
	double uy4 = uy2*uy2;
	double vx = sx+1;
	double vx2 = vx*vx;
	double vx3 = vx2*vx;
	double vy = sy+1;
	double vy2 = vy*vy;
	double vy3 = vy2*vy;
	double vy4 = vy2*vy2;
	double sy2 = sy*sy;
	double sx2 = sx*sx;
	double eps2 = eps*eps;
	double inveps2 = 1/eps2;
	double delta = eps-1;
	double delta2 = delta*delta;
	double delta3 = delta2*delta;
	double c0 = square(ux+uy);
	double c1 = c0+4*ux*uy*delta;
	assert( c1 > 0);
	assert( sx + sy >= 0); // IFE Convention
	double lEps = log(eps);
	double k = mu -(-l+l*lEps);
	double bEps = sqrt(c1);
	double d = 2*bEps*ux2*uy2*eps2;
	double psi0=0, f1=0, f2=0, psi11=0, psi12=0, psi22=0;
    using std::abs;
    using Math::signum;


	if(abs(ux) < 1e-10) {
		psi0 = (inveps2*signum(uy)*(k*uy*(3 + 6*delta2 + 2*uy + 4*delta*vy) +
		   signum(uy)*(bEps*k*(3 + 4*delta + (4 + 8*delta)*uy) +
		     2*uy*(k*(3 + 8*delta - 2*eps2*lEps + 2*delta*uy*(3 + uy) +
		         uy*(3 + 2*uy) + delta2*(6 + uy*vy)) -
		       eps*l*(uy2 + delta*(2 + uy*vy) - eps*lEps*(2 + uy*vy) +
		         eps*square(lEps))))))/(4*uy);
	} else if (abs(uy) < 1e-10) {
		psi0 = (inveps2*signum(ux)*(k*ux*(3 + 6*delta2 + 2*ux + 4*delta*vx) +
		   signum(ux)*(bEps*k*(3 + 4*delta + (4 + 8*delta)*ux) +
		     2*ux*(k*(3 + 8*delta - 2*eps2*lEps + 2*delta*ux*(3 + ux) +
		         ux*(3 + 2*ux) + delta2*(6 + ux*vx)) -
		       eps*l*(ux2 + delta*(2 + ux*vx) - eps*lEps*(2 + ux*vx) +
		         eps*square(lEps))))))/(4*ux);

	} else {
		psi0 = (bEps*(-2*eps2*lEps*ux2*uy2*(2*k + l*lEps - l*(2 + ux*vx + uy*vy)) +
		   k*(bEps*c1*(ux + uy + 2*ux*uy) +
		     ux4*(1 + 2*uy*(1 + (2 + delta*(2 + delta))*uy)) +
		     2*ux3*uy*(2 + 3*delta + uy*(3 + 2*delta*(3 + delta + uy))) +
		     uy4 + 2*ux*uy3*(3*delta + vy) + 2*ux2*uy2*
		      (3 + 8*delta + 2*delta*uy*(3 + uy) + uy*(3 + 2*uy) +
		       delta2*(6 + uy*vy))) - 2*eps*l*ux2*uy2*
		    (delta*(2 + ux*vx + uy*vy) + square(ux - uy))))/(2*d);
	}

	if(abs(ux) < 1e-10) {
		f1 =  (sx+sy<1e-10)?(1e10):(inveps2*signum(uy)*(bEps*k*(sx + sy)*uy*(-24*(1 + 2*delta) -
		     uy*(44 + 3*uy*(9 + 2*uy)) + 2*delta2*
		      (-8 + (3 + uy)*(4 + uy)*uy2) + delta*uy*(-74 - 31*uy + 2*uy3) +
		     2*delta3*sy*vy2) - (2*bEps*eps*l*(sx + sy)*uy3*vy3 +
		     k*(uy*(-(uy*(80 + uy*(192 + uy*(162 + uy*(58 + 7*uy))))) -
		         bEps*(32 + uy*(80 + uy*(96 + uy*(67 + uy*(25 + 4*uy))))) +
		         delta*vy*(sy*uy*(-60 + uy*(-26 + uy*(13 + 8*uy))) +
		           bEps*(-12 + uy*(-12 + uy*(5 + 2*uy*(4 + uy))*vy))) +
		         4*delta2*uy*(-2 + uy*(4 + uy*(7 + 2*uy)))*vy2) +
		       ux*(-2 + uy*(-2 + 5*delta + uy*(-7 + delta*(-3 - 4*delta + 8*
		                eps*uy))) + bEps*(1 - uy - 3*delta*uy + 2*delta*uy3))*
		        vy3))*signum(uy)))/(2*bEps*(sx + sy)*uy3*vy3);
	} else if (abs(uy) < 1e-10) {
		f1 =  (sx+sy<1e-10)?(1e10):(inveps2*(bEps*k*(sx + sy)*ux*(-24*(1 + 2*delta) -
		     ux*(44 + 3*ux*(9 + 2*ux)) + 2*delta2*
		      (-8 + (3 + ux)*(4 + ux)*ux2) + delta*ux*(-74 - 31*ux + 2*ux3) +
		     2*delta3*sx*vx2) -
		   (k*(-((80 + ux*(192 + ux*(162 + ux*(58 + 7*ux))))*ux2) +
		       4*delta2*ux2*(-2*sy + ux*(4 + 3*uy + ux*(7 + 2*ux + 2*uy)))*
		        vx2 + delta*ux*vx*(sx*ux*(-60 + ux*(-26 + ux*(13 + 8*ux))) +
		         (5 + ux*(-3 + 8*ux))*uy*vx2) - (2 + ux*(2 + 7*ux))*uy*vx3) +
		     bEps*(2*eps*l*(sx + sy)*ux3*vx3 +
		       k*(-(ux*(32 + ux*(80 + ux*(96 + ux*(67 + ux*(25 + 4*ux)))))) +
		         delta*ux*(-12*sy + ux*(-12*sy + (sx + sy)*ux*(5 + 2*ux*
		                (4 + ux))))*vx - (-1 + ux)*uy*vx3)))/signum(ux))*
		  signum(ux))/(2*bEps*(sx + sy)*ux3*vx3);

	} else {
		f1 =  (sx+sy<1e-10)?(1e10):(c1*k*(sy*ux4 + ux3*(1 + uy*(3 + delta - 2*delta*uy)) +
		    ux2*uy*(3 + delta - 2*uy*(-2 + delta + delta*uy)) +
		    ux*(3 + delta + uy*(3 + delta + uy))*uy2 + sy*uy3) +
		  bEps*(-2*eps*l*(sx + sy)*ux3*uy3 +
		    k*(sy*ux5 + ux4*(1 + uy*(4 + 3*delta + uy - 2*delta*uy2)) +
		      sy*uy4 + ux2*uy2*(6 + 7*uy + uy2 + 3*delta*vy) +
		      ux3*uy*(4 + 3*delta + uy*(7 + 3*delta + 4*uy -
		          2*delta*uy*vy)) + ux*uy3*(3*delta*sy + vy2))))/
		 (d*(sx + sy)*ux*uy);
	}

	if(abs(ux) < 1e-10) {
		f2 = (sx+sy<1e-10)?(-1e10):(inveps2*(4*k*(-(bEps*sy2*(4 + uy*vy*(2 - 3*delta + uy*vy))) -
		     2*(-1 + delta*uy*vy)*(4 + uy*vy*(6 - delta + 3*uy*vy))) +
		   4*delta*k*(sx + sy)*uy*(1 + 3*sy*uy)*vy3 +
		   k*(sx + sy)*uy*(5 + delta + 3*uy*(4 + 3*uy))*vy3 +
		   bEps*k*(sx + sy)*
		    (1 + uy*(3 + uy*(3 + 2*(2 + delta*(2 + delta))*uy)))*vy3 +
		   2*bEps*eps2*l*(-1 + lEps)*(sx + sy)*uy3*vy3 -
		   k*(sx + sy)*(1 + 3*sy*uy)*vy4 - bEps*k*(sx + sy)*
		    (24 + 48*delta + uy*(52 + 3*uy*(15 + uy*(6 + uy))) +
		     delta*uy*(102 + uy*(89 + 6*uy*(6 + uy))) +
		     2*delta2*vy*(4 + 3*uy*vy) - 2*delta3*vy2)*abs(uy)))/
		 (2*bEps*(sx + sy)*uy3*vy3);
	} else if (abs(uy) < 1e-10) {
		f2 = (sx+sy<1e-10)?(-1e10):((9*inveps2*k)/bEps + (12*delta*inveps2*k)/bEps + 2*(1 + inveps2)*k +
		  2*l*(-1 + lEps) + (3*inveps2*k*(2 + bEps + 4*delta - ux))/
		   (bEps*ux) + (3*inveps2*k)/ux2 + (inveps2*k*(-1 + 5*delta - 3*ux))/
		   (bEps*ux2) + (inveps2*k)/ux3 - (inveps2*k*vx)/(bEps*ux3) +
		  (4*inveps2*k*(-(bEps*sx2*(4 + ux*vx*(2 - 3*delta + ux*vx))) -
		     2*(-1 + delta*ux*vx)*(4 + ux*vx*(6 - delta + 3*ux*vx))))/
		   (bEps*(sx + sy)*ux3*vx3) -
		  (inveps2*k*(24 + 48*delta + ux*(52 + 3*ux*(15 + ux*(6 + ux))) +
		     delta*ux*(102 + ux*(89 + 6*ux*(6 + ux))) +
		     2*delta2*vx*(4 + 3*ux*vx) - 2*delta3*vx2)*signum(ux))/(ux2*vx3))/
		 2;

	} else {
		f2 = (sx+sy<1e-10)?(-1e10):(c1*k*(ux + uy)*(ux*uy*(2 + delta + 3*uy) + ux2*(1 + 3*sy*uy) +
		    uy2) + bEps*(2*eps2*l*(-1 + lEps)*(sx + sy)*ux3*uy3 +
		    k*(ux3*uy*(4 + 3*delta + uy*(9 + 6*delta +
		          2*uy*(5 + 2*delta*(2 + delta) + (2 + delta*(2 + delta))*
		             uy))) +
		      ux4*(1 + uy*(3 + uy*(3 + 2*(2 + delta*(2 + delta))*uy))) +
		      ux*(4 + 3*delta + 3*uy)*uy3 + uy4 + 3*sy*ux2*uy2*
		       (2*delta + vy))))/(d*(sx + sy)*ux*uy);
	}

	if(abs(ux) < 1e-10) {
		psi11 = (inveps2*(k*uy*(uy + 2*uy*(delta + uy + 2*delta*uy) +
		     bEps*(1 + 2*uy*(1 + (2 + delta*(2 + delta))*uy))) +
		   2*bEps*eps2*l*(-1 + lEps)*uy3 + 2*bEps*delta*eps*k*
		    (-1 + delta + 3*delta2 - 2*(1 + 2*delta)*uy)*abs(uy)))/
		 (2*bEps*uy3);
	} else if (abs(uy) < 1e-10) {
		psi11 = (inveps2*((bEps*eps2*l*(-1 + lEps)*ux +
		     k*(-(delta*eps) + (1 + 2*delta + bEps*(2 + delta*(2 + delta)))*
		        ux))*ux2 + bEps*k*(delta + delta2 - ux - 2*delta*ux)*
		    abs(ux)))/(bEps*ux3);

	} else {
		psi11 = (-2*bEps*eps2*l*ux4*uy2 + 2*bEps*eps2*l*lEps*ux4*uy2 +
		  k*(ux5*(1 + 2*uy) + ux4*(bEps + uy*(1 + 2*bEps + 2*delta +
		        2*(1 + 2*delta + bEps*(2 + delta*(2 + delta)))*uy)) -
		    2*delta*eps*ux3*(1 + 2*uy)*uy2 + 3*(bEps + uy)*uy4 +
		    2*ux2*uy3*(delta*(4 + 3*delta) + 2*delta*uy + vy) +
		    ux*uy3*(uy*(7 + 12*delta + 2*uy) + 2*bEps*(3*delta + vy))))/
		 (d*ux2);
	}

	if(abs(ux) < 1e-10) {
		psi12 = (inveps2*(k*uy*(uy*(-2 + delta*(-3 - 2*delta + 4*eps*uy)) +
		     bEps*(-2 + delta*(-3 + 2*uy2))) + 2*bEps*eps*l*uy3 +
		   2*bEps*delta*eps*k*(delta - 2*uy)*abs(uy)))/(2*bEps*uy3);
	} else if (abs(uy) < 1e-10) {
		psi12 = (inveps2*(k*ux*(ux*(-2 + delta*(-3 - 2*delta + 4*eps*ux)) +
		     bEps*(-2 + delta*(-3 + 2*ux2))) + 2*bEps*eps*l*ux3 +
		   2*bEps*delta*eps*k*(delta - 2*ux)*abs(ux)))/(2*bEps*ux3);

	} else {
		psi12 = (2*bEps*eps*l*ux3*uy3 -
		  k*(2*sy*ux5 + ux4*(2*bEps*sy + uy*(4 + 7*delta +
		        (2 + 4*delta)*uy)) + ux3*uy*(bEps*(2 + 3*delta) +
		      (2 + delta*(3 + 2*delta))*uy - 2*delta*(2 + bEps + 2*delta)*
		       uy2) + ux2*(2 + delta*(3 + 2*delta) + (2 + 4*delta)*uy)*uy3 +
		    ux*(bEps*(2 + 3*delta + 2*uy) + uy*(4 + 7*delta + 2*uy))*uy3 +
		    2*(bEps + uy)*uy4))/(d*ux*uy);
	}

	if(abs(ux) < 1e-10) {
		psi22 = (inveps2*((bEps*eps2*l*(-1 + lEps)*uy +
		     k*(-(delta*eps) + (1 + 2*delta + bEps*(2 + delta*(2 + delta)))*
		        uy))*uy2 + bEps*k*(delta + delta2 - uy - 2*delta*uy)*
		    abs(uy)))/(bEps*uy3);
	} else if (abs(uy) < 1e-10) {
		psi22 = (inveps2*(k*ux*(ux + 2*ux*(delta + ux + 2*delta*ux) +
		     bEps*(1 + 2*ux*(1 + (2 + delta*(2 + delta))*ux))) +
		   2*bEps*eps2*l*(-1 + lEps)*ux3 + 2*bEps*delta*eps*k*
		    (-1 + delta + 3*delta2 - 2*(1 + 2*delta)*ux)*abs(ux)))/
		 (2*bEps*ux3);

	} else {
		psi22 = (-2*bEps*eps2*l*ux2*uy4 + 2*bEps*eps2*l*lEps*ux2*uy4 +
		  k*(ux5*(3 + 2*uy) + ux4*(3*bEps + uy*(7 + 2*bEps + 2*uy +
		        4*delta*(3 + uy))) + 2*ux3*uy*(bEps*(2 + 3*delta) +
		      uy*(2 + delta*(4 + 3*delta - 2*eps*uy))) +
		    2*ux2*(-(delta*eps) + (1 + 2*delta +
		        bEps*(2 + delta*(2 + delta)))*uy)*uy3 + (bEps + uy)*uy4 +
		    ux*(1 + 2*bEps + 2*delta + 2*uy)*uy4))/(d*uy2);
	}


    Eigen::Matrix<double,6,1> result;
    result << psi0, f1, f2, psi11, psi12, psi22;
	return  result;
}

Eigen::Matrix<double,3,1> InvertibleNeoHookeanModel::computeStressGradientComponents(
						double sx, double sy, double l,
						double mu, double eps)
{
	double ux = sx - 1;
	double uy = sy - 1;
	double ux2 = ux*ux;
	double ux3 = ux2*ux;
	double ux4 = ux2*ux2;
	double uy2 = uy*uy;
	double uy3 = uy2*uy;
	double uy4 = uy2*uy2;
	double vx = sx+1;
	double vy = sy+1;
	double eps2 = eps*eps;
	double inveps2 = 1/eps2;
	double delta = eps-1;
	double delta2 = delta*delta;
	double c0 = square(ux+uy);
	double c1 = c0+4*ux*uy*delta;
#ifdef IFE_SOLVER
	assert (c1 > 0);
	assert (sx + sy >= 0); // IFE Convention
#endif
	double lEps = log(eps);
	double k = mu -(-l+l*lEps);
	double bEps = sqrt(c1);
	double d = 2*bEps*ux2*uy2*eps2;
	double psi0=0, psi1=0, psi2=0;
    using std::abs;
    using Math::signum;


	if(abs(ux) < 1e-10) {
		psi0 = (inveps2*signum(uy)*(k*uy*(3 + 6*delta2 + 2*uy + 4*delta*vy) +
		   signum(uy)*(bEps*k*(3 + 4*delta + (4 + 8*delta)*uy) +
		     2*uy*(k*(3 + 8*delta - 2*eps2*lEps + 2*delta*uy*(3 + uy) +
		         uy*(3 + 2*uy) + delta2*(6 + uy*vy)) -
		       eps*l*(uy2 + delta*(2 + uy*vy) - eps*lEps*(2 + uy*vy) +
		         eps*square(lEps))))))/(4*uy);
	} else if (abs(uy) < 1e-10) {
		psi0 = (inveps2*signum(ux)*(k*ux*(3 + 6*delta2 + 2*ux + 4*delta*vx) +
		   signum(ux)*(bEps*k*(3 + 4*delta + (4 + 8*delta)*ux) +
		     2*ux*(k*(3 + 8*delta - 2*eps2*lEps + 2*delta*ux*(3 + ux) +
		         ux*(3 + 2*ux) + delta2*(6 + ux*vx)) -
		       eps*l*(ux2 + delta*(2 + ux*vx) - eps*lEps*(2 + ux*vx) +
		         eps*square(lEps))))))/(4*ux);

	} else {
		psi0 = (bEps*(-2*eps2*lEps*ux2*uy2*(2*k + l*lEps - l*(2 + ux*vx + uy*vy)) +
		   k*(bEps*c1*(ux + uy + 2*ux*uy) +
		     ux4*(1 + 2*uy*(1 + (2 + delta*(2 + delta))*uy)) +
		     2*ux3*uy*(2 + 3*delta + uy*(3 + 2*delta*(3 + delta + uy))) +
		     uy4 + 2*ux*uy3*(3*delta + vy) + 2*ux2*uy2*
		      (3 + 8*delta + 2*delta*uy*(3 + uy) + uy*(3 + 2*uy) +
		       delta2*(6 + uy*vy))) - 2*eps*l*ux2*uy2*
		    (delta*(2 + ux*vx + uy*vy) + square(ux - uy))))/(2*d);
	}

	if(abs(ux) < 1e-10) {
		psi1 = (inveps2*(-2*bEps*delta*eps*k*uy*(eps + uy) +
		   (2*bEps*eps*l*uy*(delta*(-1 + lEps) + lEps + uy) +
		     k*uy*(2 + delta*(5 + 4*delta) + (3 + 8*delta*eps)*uy) +
		     bEps*k*(2 + 3*delta + uy*(3 + 2*delta*(3 + delta + uy))))*
		    abs(uy)))/(2*bEps*uy*abs(uy));
	} else if (abs(uy) < 1e-10) {
		psi1 = (inveps2*(-(bEps*k*(2 + delta*(5 + 4*delta) + (2 + 4*delta)*ux)) +
		   (2*bEps*eps*l*(lEps + delta*(-1 + lEps)*sx - ux + lEps*ux) +
		     k*(2 + delta*(5 + 4*delta) + 5*(1 + 2*delta)*ux +
		       bEps*(3 + 2*delta*(3 + delta) + 2*(2 + delta*(2 + delta))*
		          ux)))*abs(ux)))/(2*bEps*abs(ux));

	} else {
		psi1 = (2*bEps*eps2*l*lEps*sx*ux3*uy2 - 2*bEps*eps*l*ux3*
		   (delta + ux + delta*ux - uy)*uy2 +
		  c1*k*(ux3*(1 + 2*uy) + ux2*uy*(eps + uy + 2*delta*uy) -
		    ux*(eps + uy)*uy2 - uy3) +
		  bEps*k*(ux4*(1 + 2*uy*(1 + (2 + delta*(2 + delta))*uy)) +
		    ux3*uy*(2 + 3*delta + uy*(3 + 2*delta*(3 + delta + uy))) - uy4 -
		    ux*uy3*(3*delta + vy)))/(d*ux);
	}

	if(abs(ux) < 1e-10) {
		psi2 = (inveps2*(-(bEps*k*(2 + delta*(5 + 4*delta) + (2 + 4*delta)*uy)) +
		   (2*bEps*eps*l*(lEps + delta*(-1 + lEps)*sy - uy + lEps*uy) +
		     k*(2 + delta*(5 + 4*delta) + 5*(1 + 2*delta)*uy +
		       bEps*(3 + 2*delta*(3 + delta) + 2*(2 + delta*(2 + delta))*
		          uy)))*abs(uy)))/(2*bEps*abs(uy));
	} else if (abs(uy) < 1e-10) {
		psi2 = (inveps2*(-2*bEps*delta*eps*k*ux*(eps + ux) +
		   (2*bEps*eps*l*ux*(delta*(-1 + lEps) + lEps + ux) +
		     k*ux*(2 + delta*(5 + 4*delta) + (3 + 8*delta*eps)*ux) +
		     bEps*k*(2 + 3*delta + ux*(3 + 2*delta*(3 + delta + ux))))*
		    abs(ux)))/(2*bEps*ux*abs(ux));

	} else {
		psi2 = (-(c1*k*(sy*ux3 + ux2*uy*(eps - uy - 2*delta*uy) -
		     ux*(eps + 2*uy)*uy2 - uy3)) + 2*bEps*eps2*l*lEps*sy*ux2*uy3 -
		  2*bEps*eps*l*ux2*(delta - ux + uy + delta*uy)*uy3 +
		  bEps*k*(-(sy*ux4) + ux3*uy*(-2 + delta*(-3 + 2*uy2)) +
		    ux*(2 + 3*delta + 2*uy)*uy3 + ux2*(3 + 2*delta*(3 + delta) +
		      2*(2 + delta*(2 + delta))*uy)*uy3 + uy4))/(d*uy);
	}


    Eigen::Matrix<double,3,1> result;
    result << psi0, psi1, psi2;
    return result;
}
