/*
 * FeasibleWrenchPolytope_API.cpp
 *
 *  Created on: April 12, 2017
 *      Author: Romeo Orsolino
 */

#include <FeasibleWrenchPolytope_API.h>

FeasibleWrenchPolytope_API::FeasibleWrenchPolytope_API() {

	g0 = rbd::g;	// Initialize g0 with the gravity vector rbd::g
	// TODO get the mass from the parameter getter
//	this->mass = robot_mass;
//	I = rbd::InertiaMatrixSparse();


}

FeasibleWrenchPolytope_API::~FeasibleWrenchPolytope_API() {
	// TODO Auto-generated destructor stub
}

void FeasibleWrenchPolytope_API::init(){

//	Initialize the solvers
//	solver.reset(new dwl::solver::IpoptNLP());
//	margin_solver.reset(new dwl::solver::IpoptNLP());

//	Initilize FWP variables
	for (int leg = iit::dog::LF; leg <= iit::dog::RH; leg++) {
		force_polytopes[leg].v_rep.resize(3,6);
		force_polytopes[leg].v_rep.setZero();
		force_polytopes[leg].hs_rep.resize(5,4);
		force_polytopes[leg].hs_rep.setZero();
		force_polytopes[leg].hs_rep.block(0,3,5,1).setConstant(-1000.0);
		linear_friction_cones[leg].resize(3,6);
		linear_friction_cones[leg].setZero();
		bounded_friction_poly[leg].v_rep.resize(3,6);
		bounded_friction_poly[leg].v_rep.setZero();


	}

	lfc_top_vec.resize(1);
	lfc_top_vec.setZero();
	lfc_top_map.resize(1,1);
	lfc_top_map.setZero();
	
	fp_top_vec.resize(1);
	fp_top_vec.setZero();
	fp_top_map.resize(1,1);
	fp_top_map.setZero();

	bfc_top_vec.resize(1);
	bfc_top_vec.setZero();
	bfc_top_map.resize(1,1);
	bfc_top_map.setZero();



	// Setting up the system from the hyq urdf file
//	model_ = ros::package::getPath(std::string("contact_wrench_set")) + "/config/hyq.urdf";
//	robot_ = ros::package::getPath(std::string("contact_wrench_set")) + "/config/hyq.yarf";
//	fbs.resetFromURDFFile(model_, robot_);
//	wb_kin.modelFromURDFFile(model_, robot_);
//	feet_names = fbs.getEndEffectorNames(dwl::model::FOOT);
//	foot_id_map_.resize(fbs.getNumberOfEndEffectors(dwl::model::FOOT));
//	foot_id_map_[dog::LF] = fbs.getEndEffectorId("lf_foot");
//	foot_id_map_[dog::RF] = fbs.getEndEffectorId("rf_foot");
//	foot_id_map_[dog::LH] = fbs.getEndEffectorId("lh_foot");
//	foot_id_map_[dog::RH] = fbs.getEndEffectorId("rh_foot");

}


void FeasibleWrenchPolytope_API::ZMP(Eigen::Vector3d r,
 						   Eigen::Vector3d r_dd,
						   Eigen::Vector3d & zmp)
{
	double height = r(rbd::Z);
	zmp = r - r_dd*height/g0;
	zmp(rbd::Z) = 0.0;
}

void FeasibleWrenchPolytope_API::ZMPstability(iit::dog::LegDataMap<rbd::Vector3d> stance_feet_WF,
						   iit::dog::LegDataMap<bool> stance_legs_flag,
						   Eigen::Vector3d r,
 						   Eigen::Vector3d r_dd,
						   Eigen::Vector3d & zmp,
						   double & ZMPdistance)
{
	ZMP(r, r_dd, zmp);
	double dist[4];
//	double min;
	if((stance_legs_flag[dog::LF])&&(stance_legs_flag[dog::RF])&&(stance_legs_flag[dog::LH])&&(stance_legs_flag[dog::RH]))
	{
		distPointToLine(stance_feet_WF[dog::LF],stance_feet_WF[dog::RF], zmp, dist[0]);
		distPointToLine(stance_feet_WF[dog::RF],stance_feet_WF[dog::RH], zmp, dist[1]);
		distPointToLine(stance_feet_WF[dog::RH],stance_feet_WF[dog::LH], zmp, dist[2]);
		distPointToLine(stance_feet_WF[dog::LH],stance_feet_WF[dog::LF], zmp, dist[3]);
//		std::cout<<dist[0]<<" "<<dist[1]<<" "<<dist[2]<<" "<<dist[3]<<" "<<std::endl;
		ZMPdistance = *std::min_element(std::begin(dist), std::end(dist));
	}else{
			int triangle_count=0;
			Eigen::Vector3d triangle_pnt[3], edges[3];
			for (int i = 0; i<dog::_LEGS_COUNT; i++)
			{
				if (stance_legs_flag[i])
				{
					triangle_pnt[triangle_count] = stance_feet_WF[i];
					triangle_count++;
				}
			}
//			std::cout<<"triple stance"<<std::endl;
			distPointToLine(triangle_pnt[0],triangle_pnt[1], zmp, dist[1]);
			distPointToLine(triangle_pnt[1],triangle_pnt[2], zmp, dist[2]);
			distPointToLine(triangle_pnt[2],triangle_pnt[0], zmp, dist[3]);
	//		std::cout<<dist[0]<<" "<<dist[1]<<" "<<dist[2]<<" "<<dist[3]<<" "<<std::endl;
			ZMPdistance = *std::min_element(std::begin(dist), std::end(dist));
	}


}

void FeasibleWrenchPolytope_API::distPointToLine(Eigen::Vector3d P1,
		Eigen::Vector3d P2,
		Eigen::Vector3d X,
		double & dist)
{//		implemented as in the website: http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
	Eigen::Vector3d dist_num_vec = (X-P1).cross(X-P2);
	Eigen::Vector3d dist_denum_vec = (P2-P1);
	double dist_num = sqrt(pow((double)dist_num_vec(rbd::X),2)+pow((double)dist_num_vec(rbd::Y),2)+pow((double)dist_num_vec(rbd::Z),2));
	double dist_denum = sqrt(pow((double)dist_denum_vec(rbd::X),2)+pow((double)dist_denum_vec(rbd::Y),2)+pow((double)dist_denum_vec(rbd::Z),2));
	dist = dist_num/dist_denum;
//	std::cout<<P1<<" "<<P2<<" "<<X<<std::endl;

}

void FeasibleWrenchPolytope_API::IntertialWrench_BF(Eigen::Vector3d r_d,
												   Eigen::Vector3d r_dd,
												   Eigen::Vector3d orient_d,
												   Eigen::Vector3d orient_dd,
												   iit::rbd::Matrix66d InertiaMatrix,
												   iit::rbd::ForceVector & hdot_BF)
{

//	TODO: I should properly compute the angular velocity omega and acceleration omega_dot starting
//	Eigen::Matrix3d R = put::math::getRotationMatrix(orient_d);
//	Eigen::Vector3d omega = R*orient_d;
	vel.head(3) = orient_d;
	vel.tail(3) = r_d;

	acc.head(3) = orient_dd;
	Eigen::Vector3d bias_acc;
	Eigen::Matrix3d omega_cross = iit::rbd::Utils::buildCrossProductMatrix(orient_d);
	bias_acc = omega_cross * r_d;
	acc.tail(3) = r_dd - bias_acc;

	Eigen::Matrix<double, 6,1> sp_hdot, sp_acc;

	RigidBodyDynamics::Math::SpatialMatrix tmp_bias_wrench;
	iit::rbd::ForceVector bias_wrench;

	RigidBodyDynamics::Math::SpatialMatrix vel_crossf = RigidBodyDynamics::Math::crossf(vel);
	tmp_bias_wrench = vel_crossf * InertiaMatrix;
	bias_wrench = tmp_bias_wrench * vel;
	hdot_BF = InertiaMatrix*acc + bias_wrench; // equation of motion

}

void FeasibleWrenchPolytope_API::IntertialWrench_WF(Eigen::Vector3d r,
	                                               Eigen::Vector3d r_d,
	                                               Eigen::Vector3d r_dd,
	                                               Eigen::Vector3d orient,
	                                               Eigen::Vector3d orient_d,
	                                               Eigen::Vector3d orient_dd,
	                                               iit::rbd::Matrix66d InertiaMatrix,
												   iit::rbd::ForceVector & hdot_WF){


		iit::rbd::ForceVector hdot_BF;
		IntertialWrench_BF(r_d,
		                                 r_dd,
		                                 orient_d,
		                                 orient_dd,
										 InertiaMatrix,
		                                 hdot_BF);
	//	rotation from base frame to horizontal frame using spatial transformations
		double rot_x, rot_y, rot_z;
		rot_x = orient(rbd::X);
		rot_y = orient(rbd::Y);
		rot_z = orient(rbd::Z);
		RigidBodyDynamics::Math::SpatialTransform sp_tr_x, sp_tr_y, sp_tr_z;
		sp_tr_x = RigidBodyDynamics::Math::Xrotx(rot_x);
		sp_tr_y = RigidBodyDynamics::Math::Xroty(rot_y);
		sp_tr_z = RigidBodyDynamics::Math::Xrotz(rot_z);
		RigidBodyDynamics::Math::SpatialMatrix tr_mat_x, tr_mat_y, tr_mat_z, tr_mat;
		tr_mat_x  = sp_tr_x.toMatrix();
		tr_mat_y  = sp_tr_y.toMatrix();
		tr_mat_z  = sp_tr_z.toMatrix();
		iit::rbd::ForceVector hdot_HF = tr_mat_x*tr_mat_y*tr_mat_z*hdot_BF;

	//	translation from horizontal to world frame using spatial transformation
		RigidBodyDynamics::Math::SpatialTransform X_trasl;
		X_trasl = RigidBodyDynamics::Math::Xtrans(r);
		RigidBodyDynamics::Math::SpatialMatrix X_trasl_mat;
		X_trasl_mat = X_trasl.toMatrix();
		inertial_wrench_WF = X_trasl_mat*hdot_HF;
		hdot_WF = inertial_wrench_WF;
}

void FeasibleWrenchPolytope_API::GravitationalWrench_WF(Eigen::Vector3d com_posWF, CWSData cws_struct){

    //compute wrench expressed in  world frame
    iit::rbd::ForceVector grav_wrench_com;
    grav_wrench_com.setZero();
    grav_wrench_com(rbd::LZ) =  - cws_struct.robot_mass *g0;
    rbd::angularPart(grav_wrench_WF) = com_posWF.cross(rbd::linearPart(grav_wrench_com));
    rbd::linearPart(grav_wrench_WF) = rbd::linearPart(grav_wrench_com);
//	setGW_WF(grav_wrench_WF);
}

void FeasibleWrenchPolytope_API::GravitationalWrench_WF(Eigen::Vector3d com_posWF, 
															CWSData cws_struct, 
															Eigen::VectorXd & grav_wrench_W){

	GravitationalWrench_WF(com_posWF, cws_struct);
    grav_wrench_W = grav_wrench_WF;

}

//void FeasibleWrenchPolytope_API::compute_cddlib(CWSData cws_struct, Eigen::MatrixXd & A_hs_description){
//
//	compute_cddlib(cws_struct.stance_feet_WF,
//								cws_struct.stance_legs_flag,
//			                    cws_struct.friction,
//								cws_struct.normal,
//								A_hs_description);
//
//}

//void FeasibleWrenchPolytope_API::compute_cddlib(iit::dog::LegDataMap<rbd::Vector3d> stance_feet_WF,
//							 iit::dog::LegDataMap<bool> stance_legs_flag,
//		                     iit::dog::LegDataMap<double> friction_coeff,
//		                     iit::dog::LegDataMap<rbd::Vector3d> terrain_normal,
//							 Eigen::MatrixXd & A_hs_description){
//
//
//	/* convert the friction cone coeff, the feet positions and the terrain normals
//	 * into the edges of the 4 6-dimensional polytopes/polyhedral cones
//	 */
////	iit::dog::LegDataMap< Eigen::MatrixXd > legs_cone;
////	CreateEdges(stance_feet_WF, friction_coeff, terrain_normal, legs_cone);
//
//	/*	Now convert the force vectors into the generator_Rn type */
//
//	ddf_PolyhedraPtr poly;
//	ddf_LPPtr lp;
//	ddf_ErrorType err=ddf_NoError;
////	FILE *reading=NULL, *writing;
//	myfloat val;
//	time_t starttime,endtime;
//	ddf_rowset redrows,linrows;
//
//	ddf_NumberType numb;
//	ddf_rowrange m;
//	ddf_colrange n;
//	ddf_MatrixPtr M = NULL, A= NULL;
//	ddf_MatrixPtr M_LF = NULL;
//	ddf_MatrixPtr M_LH = NULL;
//	ddf_MatrixPtr M_RF = NULL;
//	ddf_MatrixPtr M_RH = NULL;
//
//	ddf_set_global_constants();  /* First, this must be called. */
//
//	numb = ddf_Real;   /* set a number type */
//	m=4;    /* number of rows, number of edges  */
//	n=7;    /* number of columns, dimension of the space */
//	//A=ddf_CreateMatrix(0,7);
//	M=ddf_CreateMatrix(0,n);
//	err=ddf_NoError;
//
//	ddf_init(val);
//
//	stance_num = 0;
//	margin = 1000000.0;
//	double cws_feasibility;
//// std::cout<<"starting computation"<<std::endl;
//	if(stance_legs_flag[dog::LF]){
//		//M_LF=ddf_CreateMatrix(m,n);
//		M_LF = eigen2cdd(cws_pyramid[dog::LF]);
//		M_LF->representation=ddf_Generator;
////		ddf_WriteMatrix(stdout,M_LF);
//		M = ddf_AppendMatrix(M, M_LF);
//		stance_num+=1;
//		if (err!=ddf_NoError) {
//			std::cout<<"error LF"<<std::endl;
//			ddf_WriteErrorMessages(stdout,err);
//			cws_feasibility = old_cws_feasibility;
//			goto _L99;
//		}
//	}
//	if(stance_legs_flag[dog::LH]){
//		//M_LH=ddf_CreateMatrix(m,n);
//		M_LH = eigen2cdd(cws_pyramid[dog::LH]);
//		M_LH->representation=ddf_Generator;
////		ddf_WriteMatrix(stdout,M_LH);
//		M = ddf_AppendMatrix(M, M_LH);
//		stance_num+=1;
//		if (err!=ddf_NoError) {
//			std::cout<<"error LH"<<std::endl;
//			ddf_WriteErrorMessages(stdout,err);
//			cws_feasibility = old_cws_feasibility;
//			goto _L99;
//		}
//	}
//	if(stance_legs_flag[dog::RF]){
//		//M_RF=ddf_CreateMatrix(m,n);
//		M_RF = eigen2cdd(cws_pyramid[dog::RF]);
//		M_RF->representation=ddf_Generator;
////		ddf_WriteMatrix(stdout,M_RF);
//		M = ddf_AppendMatrix(M, M_RF);
//		stance_num+=1;
//		if (err!=ddf_NoError) {
//			std::cout<<"error RF"<<std::endl;
//			ddf_WriteErrorMessages(stdout,err);
//			cws_feasibility = old_cws_feasibility;
//			goto _L99;
//		}
//	}
//	if(stance_legs_flag[dog::RH]){
//		//M_RH=ddf_CreateMatrix(m,n);
//		M_RH = eigen2cdd(cws_pyramid[dog::RH]);
//		M_RH->representation=ddf_Generator;
//		M = ddf_AppendMatrix(M, M_RH);
////		ddf_WriteMatrix(stdout,M_RH);
//		stance_num+=1;
//		if (err!=ddf_NoError) {
//			std::cout<<"error RH"<<std::endl;
//			ddf_WriteErrorMessages(stdout,err);
//			cws_feasibility = old_cws_feasibility;
//			goto _L99;
//		}
//	}
////	  std::cout<<"remove redundancy"<<std::endl;
////	ddf_MatrixCanonicalize(&M, &impl_lin, &redset, &newpos, &err);
////	ddf_WriteMatrix(stdout,M);
//	poly=ddf_DDMatrix2Poly(M, &err);
////	 std::cout<<"DD computed"<<std::endl;
//	if (err!=ddf_NoError) {
////		ddf_WriteErrorMessages(stdout,err);
//		cws_feasibility = old_cws_feasibility;
////		std::cout<<"error in the double description computation"<<std::endl;
//		A_hs_description.resize(0,0);
//		goto _L99;
//	}
//	if (err==ddf_NoError) {
//		A=ddf_CopyInequalities(poly);
////		A->representation=ddf_Inequality;
//		ddf_MatrixCanonicalize(&A, &impl_lin, &redset, &newpos, &err);
//		//poly=ddf_DDMatrix2Poly(A, &err);  /* compute the second (generator) representation */
//		if (err!=ddf_NoError) {
//			ddf_WriteErrorMessages(stdout,err);
//			cws_feasibility = old_cws_feasibility;
//			goto _L99;
//		}
////		ddf_WriteMatrix(stdout,A);
//		A_hs_description.resize(A->rowsize,A->colsize);
//		cdd2eigen(A,A_hs_description);
//	}else{
////		ddf_WriteErrorMessages(stdout,err);
////		dist = -2;
//	}
//
//	_L99:
//	ddf_free_global_constants();  /* At the end, this must be called. */
//}


void FeasibleWrenchPolytope_API::contact_wrench_set(const CWSData cws_struct, Eigen::MatrixXd & A_v_description){
	MS_LP ms_lp;ms_lp.setPrintAll(print_all);
	Eigen::MatrixXd M1;
	Eigen::MatrixXd M2;
	Eigen::MatrixXd sum1;
	Eigen::MatrixXd sum2;
	// two cubes
//	M1.resize(6,8);
//	M2.resize(6,8);
//	M1(0,0) = 0.0;	  M1(1,0) = -10.0;    M1(2,0) = 0.0;	  M1(3,0) = 0.0;	  M1(4,0) = 0.0;      M1(5,0) = 0.0;
//	M1(0,1) = 10.0;	  M1(1,1) = 0.0;      M1(2,1) = 0.0;	  M1(3,1) = 0.0;	  M1(4,1) = 0.0;      M1(5,1) = 0.0;
//	M1(0,2) = -10.0;  M1(1,2) = 0.0;      M1(2,2) = 0.0;	  M1(3,2) = 0.0;	  M1(4,2) = 0.0;      M1(5,2) = 0.0;
//	M1(0,3) = 0.0;	  M1(1,3) = 10.0;     M1(2,3) = 0.0;	  M1(3,3) = 0.0;	  M1(4,3) = 0.0;      M1(5,3) = 0.0;
//	M1(0,4) = 0.0;	  M1(1,4) = -10.0;    M1(2,4) = 10.0;	  M1(3,4) = 0.0;	  M1(4,4) = 0.0;      M1(5,4) = 0.0;
//	M1(0,5) = 10.0;	  M1(1,5) = 0.0;      M1(2,5) = 10.0;	  M1(3,5) = 0.0;	  M1(4,5) = 0.0;      M1(5,5) = 0.0;
//	M1(0,6) = -10.0;  M1(1,6) = 0.0;      M1(2,6) = 10.0;	  M1(3,6) = 0.0;	  M1(4,6) = 0.0;      M1(5,6) = 0.0;
//	M1(0,7) = 0.0;	  M1(1,7) = 10.0;     M1(2,7) = 10.0;	  M1(3,7) = 0.0;	  M1(4,7) = 0.0;      M1(5,7) = 0.0;
//
//	M2(0,0) = 0.0;	  M2(1,0) = -10.0;    M2(2,0) = 0.0;	  M2(3,0) = 0.0;	  M2(4,0) = 0.0;      M2(5,0) = 0.0;
//	M2(0,1) = 10.0;	  M2(1,1) = 0.0;      M2(2,1) = 0.0;	  M2(3,1) = 0.0;	  M2(4,1) = 0.0;      M2(5,1) = 0.0;
//	M2(0,2) = -10.0;  M2(1,2) = 0.0;      M2(2,2) = 0.0;	  M2(3,2) = 0.0;	  M2(4,2) = 0.0;      M2(5,2) = 0.0;
//	M2(0,3) = 0.0;	  M2(1,3) = 10.0;     M2(2,3) = 0.0;	  M2(3,3) = 0.0;	  M2(4,3) = 0.0;      M2(5,3) = 0.0;
//	M2(0,4) = 0.0;	  M2(1,4) = -10.0;    M2(2,4) = 10.0;	  M2(3,4) = 0.0;	  M2(4,4) = 0.0;      M2(5,4) = 0.0;
//	M2(0,5) = 10.0;	  M2(1,5) = 0.0;      M2(2,5) = 10.0;	  M2(3,5) = 0.0;	  M2(4,5) = 0.0;      M2(5,5) = 0.0;
//	M2(0,6) = -10.0;  M2(1,6) = 0.0;      M2(2,6) = 10.0;	  M2(3,6) = 0.0;	  M2(4,6) = 0.0;      M2(5,6) = 0.0;
//	M2(0,7) = 0.0;	  M2(1,7) = 10.0;     M2(2,7) = 10.0;	  M2(3,7) = 0.0;	  M2(4,7) = 0.0;      M2(5,7) = 0.0;


//// test with friction cones (Fz, Fx, Tau y)
////	M1.resize(3,3);
////	M2.resize(3,3);
//	//	Fz				Fx 		Tau y  = Fx *foot_z - Fz* foot_x (for foot = [1,1] = [foot_x, foot_z])
//	M1(0,0) = 0.0;	  M1(1,0) = 0.0;      M1(2,0) = 0.0;
//	M1(0,1) = 10.0;	  M1(1,1) = 10.0;     M1(2,1) = 0.0;  // 1*10 - 1*10 = 0
//	M1(0,2) = 10.0;	  M1(1,2) = -10.0;    M1(2,2) = -20.0;  // 1* -10 - 1*10 = -20
//
//	//	Fz				Fx 		Tau y  = Fx *foot_z - Fz* foot_x (for foot = [2,1] = [foot_x, foot_z])
//	M2(0,0) = 0.0;	  M2(1,0) = 0.0;      M2(2,0) = 0.0;
//	M2(0,1) = 10.0;	  M2(1,1) = 10.0;     M2(2,1) = -10.0; // 10*1 - 10*2 =  - 10
//	M2(0,2) = 10.0;	  M2(1,2) = -10.0;    M2(2,2) = -30.0;  // -10*1 - 10*2 =   -30


	// two pyramids
//	M1.resize(6,5);
//	M2.resize(6,5);
//	M1(0,0) = 0.0;	  M1(1,0) = 0.0;      M1(2,0) = 0.0;	  M1(3,0) = 0.0;	  M1(4,0) = 0.0;      M1(5,0) = 0.0;
//	M1(0,1) = 10.0;	  M1(1,1) = 0.0;      M1(2,1) = 10.0;	  M1(3,1) = 0.0;	  M1(4,1) = 0.0;      M1(5,1) = 0.0;
//	M1(0,2) = 0.0;	  M1(1,2) = 10.0;     M1(2,2) = 10.0;	  M1(3,2) = 0.0;	  M1(4,2) = 0.0;      M1(5,2) = 0.0;
//	M1(0,3) = 0.0;	  M1(1,3) = -10.0;    M1(2,3) = 10.0;	  M1(3,3) = 0.0;	  M1(4,3) = 0.0;      M1(5,3) = 0.0;
//	M1(0,4) = -10.0;  M1(1,4) = 0.0;      M1(2,4) = 10.0;	  M1(3,4) = 0.0;	  M1(4,4) = 0.0;      M1(5,4) = 0.0;
////	M1(0,5) = 2.0;	  M1(1,5) = 0.0;      M1(2,5) = 1.0;	  M1(3,5) = 0.0;	  M1(4,5) = 0.0;      M1(5,5) = 0.0;
////	M1(0,6) = 0.0;	  M1(1,6) = 2.0;      M1(2,6) = 1.0;	  M1(3,6) = 0.0;	  M1(4,6) = 0.0;      M1(5,6) = 0.0;
////	M1(0,7) = 2.0;	  M1(1,7) = 2.0;      M1(2,7) = 1.0;	  M1(3,7) = 0.0;	  M1(4,7) = 0.0;      M1(5,7) = 0.0;
//
//	M2(0,0) = 0.0;	  M2(1,0) = 0.0;      M2(2,0) = 0.0;	  M2(3,0) = 0.0;	  M2(4,0) = 0.0;      M2(5,0) = 0.0;
//	M2(0,1) = 10.0;	  M2(1,1) = 0.0;      M2(2,1) = 10.0;	  M2(3,1) = 0.0;	  M2(4,1) = 0.0;      M2(5,1) = 0.0;
//	M2(0,2) = 0.0;	  M2(1,2) = 10.0;     M2(2,2) = 10.0;	  M2(3,2) = 0.0;	  M2(4,2) = 0.0;      M2(5,2) = 0.0;
//	M2(0,3) = 0.0;	  M2(1,3) = -10.0;    M2(2,3) = 10.0;	  M2(3,3) = 0.0;	  M2(4,3) = 0.0;      M2(5,3) = 0.0;
//	M2(0,4) = -10.0;  M2(1,4) = 0.0;      M2(2,4) = 10.0;	  M2(3,4) = 0.0;	  M2(4,4) = 0.0;      M2(5,4) = 0.0;
////	M2(0,5) = 2.0;	  M2(1,5) = 0.0;      M2(2,5) = 1.0;	  M2(3,5) = 0.0;	  M2(4,5) = 0.0;      M2(5,5) = 0.0;

	// test with friction cones (Fz, Fx, Tau y)
//	M1.resize(6,3);
//	M2.resize(6,3);
//	//	Fz				Fx 		Tau y  = Fx *foot_z - Fz* foot_x (for foot = [1,1] = [foot_x, foot_z])
//	M1(0,0) = 0.0;	  M1(1,0) = 0.0;      M1(2,0) = 0.0; 		M1(3,0) = 0.0;	  M1(4,0) = 0.0;      M1(5,0) = 0.0;
//	M1(0,1) = 0.0;	  M1(1,1) = -80.0;    M1(2,1) = 0.0;        M1(3,1) = -20.0;  M1(4,1) = 0.0;      M1(5,1) = 20.0;
//	M1(0,2) = 0.0;	  M1(1,2) = -40.0;    M1(2,2) = 0.0;        M1(3,2) = 20.0;	  M1(4,2) = 0.0;      M1(5,2) = 20.0;
//
//	//	Fz				Fx 		Tau y  = Fx *foot_z - Fz* foot_x (for foot = [2,1] = [foot_x, foot_z])
//	M2(0,0) = 0.0;	  M2(1,0) = 0.0;      M2(2,0) = 0.0; 		M2(3,0) = 0.0;	  M2(4,0) = 0.0;      M2(5,0) = 0.0;
//	M2(0,1) = 0.0;	  M2(1,1) = -40.0;    M2(2,1) = 0.0;        M2(3,1) = -20.0;  M2(4,1) = 0.0;      M2(5,1) = 20.0;
//	M2(0,2) = 0.0;	  M2(1,2) = 0.0;      M2(2,2) = 0.0;        M2(3,2) = 20.0;	  M2(4,2) = 0.0;      M2(5,2) = 20.0;

	iit::dog::LegDataMap< Eigen::MatrixXd > legs_cone;
	CreateEdges(cws_struct.stance_feet_WF,
				cws_struct.friction,
				cws_struct.normal,
				cws_struct.max_normal_force,
				legs_cone);


	clock_t start = clock();
	ms_lp.mink_sum_with_vertices(legs_cone[dog::LH], legs_cone[dog::LF], sum1);
//	ms_lp.mink_sum_with_vertices(M1, M2, A_v_description);
//	std::cout<<"first sum: "<<sum1<<std::endl;
	ms_lp.mink_sum_with_vertices(legs_cone[dog::RF], legs_cone[dog::RH] , sum2);
////	std::cout<<"second sum: "<<sum2<<std::endl;
	ms_lp.mink_sum_with_vertices(sum2, sum1, A_v_description);
	std::cout<<"size of v description: "<<A_v_description.rows() <<" x "<<A_v_description.cols()<<std::endl;
//	std::cout<<"third sum: "<<A_v_description<<std::endl;
	clock_t end = clock();
	float seconds = (float)(end - start) / CLOCKS_PER_SEC;
	printf ("compute politopix seconds: %f  \n", seconds);
}

void FeasibleWrenchPolytope_API::get_hs_description_politopix(Eigen::MatrixXd & A_v_description, boost::shared_ptr<Polytope_Rn> & primal_polytope){

	unsigned int dim = A_v_description.rows(); // dimensionality of each vertex
	unsigned int vx_num = A_v_description.cols(); // num of vertices
	double margin_tol = 10e-7;
	Rn::setDimension(dim);
	Rn::setTolerance(margin_tol);
//	boost::shared_ptr<Polytope_Rn> primal_polytope(new Polytope_Rn());
	for (int vert = 0; vert< vx_num; vert++){
		boost::shared_ptr<Generator_Rn> VX(new Generator_Rn(dim));
		for (unsigned int j = 0; j< dim; j++){
			VX->setCoordinate(j, round(A_v_description(j,vert)));
		}
		primal_polytope->addGenerator(VX);
	}

	// Compute the double description removing the interior point.
	politopixAPI::computeDoubleDescription(primal_polytope,1000.);
	bool check_topology = politopixAPI::checkTopologyAndGeometry(primal_polytope);
//	std::cout<<"topology check: "<<check_topology<<std::endl;
	reorder_v_description(primal_polytope, A_v_description);
}

void FeasibleWrenchPolytope_API::get_hs_description_politopix(Eigen::MatrixXd & A_v_description, Eigen::MatrixXd & A_hs_description){

	unsigned int dim = A_v_description.rows(); // dimensionality of each vertex
	unsigned int vx_num = A_v_description.cols(); // num of vertices
	double margin_tol = 10e-7;
	Rn::setDimension(dim);
	Rn::setTolerance(margin_tol);
	boost::shared_ptr<Polytope_Rn> primal_polytope(new Polytope_Rn());
	get_hs_description_politopix(A_v_description, primal_polytope);
	eigen_hs_description(primal_polytope, A_hs_description);
}

void FeasibleWrenchPolytope_API::eigen_hs_description(const boost::shared_ptr<Polytope_Rn> primal_polytope, Eigen::MatrixXd & A_hs_description){
	unsigned int hs_size = primal_polytope->numberOfHalfSpaces();
	unsigned int dim = primal_polytope->dimension();
	A_hs_description.resize(hs_size,dim+1);
	constIteratorOfListOfGeometricObjects< boost::shared_ptr<HalfSpace_Rn> > hs_iterator(primal_polytope->getListOfHalfSpaces());
	unsigned int i = 0;
	for (hs_iterator.begin(); hs_iterator.end()!=true; hs_iterator.next()) {
		const boost::shared_ptr<HalfSpace_Rn>& currentHalfSpace = hs_iterator.current();
//		std::cout<< currentHalfSpace->dimension()<<std::endl;
		for (unsigned int j = 0; j< dim; j++){
			A_hs_description(i,j) = currentHalfSpace->getCoefficient(j);
		}
//		The known term b is in the last column!!!
		A_hs_description(i,dim) = currentHalfSpace->getConstant();
		i++;
	}
}

void FeasibleWrenchPolytope_API::eigen_v_description(const boost::shared_ptr<Polytope_Rn> primal_polytope, Eigen::MatrixXd & A_v_description){
	unsigned int v_size = primal_polytope->numberOfGenerators();
	unsigned int dim = primal_polytope->dimension();
	A_v_description.resize(dim,v_size);
	constIteratorOfListOfGeometricObjects< boost::shared_ptr<Generator_Rn> > v_iterator(primal_polytope->getListOfGenerators());
	unsigned int i = 0;
	for (v_iterator.begin(); v_iterator.end()!=true; v_iterator.next()) {
		const boost::shared_ptr<Generator_Rn>& currentGenerator = v_iterator.current();
//		std::cout<< currentHalfSpace->dimension()<<std::endl;
		for (unsigned int j = 0; j< dim; j++){
			A_v_description(j,i) = currentGenerator->getCoordinate(j);
			A_v_description(j,i) = round(A_v_description(j,i));
		}
		i++;
	}
}


void FeasibleWrenchPolytope_API::hs_based_margin(const FeasibleWrenchPolytope_API::wrenchType wrench_type,
									const Eigen::MatrixXd M1,
									const Eigen::VectorXd wrench_gi,
									double & fwp_hs_margin){

    switch(wrench_type){
    case full_6D:
    	std::cout<<"I am HERE"<<std::endl;
    	FeasibilityMargin(wrench_type, M1, wrench_gi, fwp_hs_margin);
     	break;
    case angular_3D:
    	FeasibilityMargin(wrench_type, M1, wrench_gi, fwp_hs_margin);
    	break;
    case linear_3D:
    	FeasibilityMargin(wrench_type, M1, wrench_gi, fwp_hs_margin);
    	break;
    default:
    	break;
    }

}

void FeasibleWrenchPolytope_API::FeasibilityMargin(const FeasibleWrenchPolytope_API::wrenchType wrench_type,
									const Eigen::MatrixXd M1,
									const Eigen::VectorXd wrench_gi,
									double & cws_feasibility){
	////////////////////////////////////////////
	//	Compute distance point to hyperplane  //
	////////////////////////////////////////////
	rbd::ForceVector p;
	Eigen::MatrixXd M, M_tmp;

	margin = 50000;
	unsigned int rows;

    switch(wrench_type){
    case full_6D:
    	rows = 6;
    	p.resize(rows);
		p = wrench_gi;
		M_tmp = M1;

     	break;
    case angular_3D:
    	rows = 3;
    	p.resize(rows);
		p.setZero();
		rbd::linearPart(p) = wrench_gi;
		M_tmp.resize(M1.rows(),7);
		M_tmp.setZero();
		M_tmp.block(0,1,M1.rows(),3) = M1.block(0,0,M1.rows(),3);
		M_tmp.block(0,0,M1.rows(),1) = M1.block(0,M1.cols()-1,M1.rows(),1);

    	break;

    case linear_3D:
    	rows = 3;
    	p.resize(rows);
		p.setZero();
		rbd::linearPart(p) = wrench_gi;
		M_tmp.resize(M1.rows(),7);
		M_tmp.setZero();
		M_tmp.block(0,4,M1.rows(),3) = M1.block(0,0,M1.rows(),3);
		M_tmp.block(0,0,M1.rows(),1) = M1.block(0,M1.cols()-1,M1.rows(),1);

    	break;
    default:
    	break;
    }

//	std::cout<<"feasibility wrench: "<<wrench_gi.transpose()<<std::endl;
//	if(wrench_gi.rows()==6){
//		std::cout<<"wrench is 6D"<<std::endl;
//
//	}else{
//		p.setZero();
//		rbd::linearPart(p) = wrench_gi;
//		M_tmp.resize(M1.rows(),7);
//		M_tmp.setZero();
//		M_tmp.block(0,4,M1.rows(),3) = M1.block(0,0,M1.rows(),3);
//		M_tmp.block(0,0,M1.rows(),1) = M1.block(0,M1.cols()-1,M1.rows(),1);
//	}
	// Safety check
	if (wrench_gi.rows() != M_tmp.cols()){
		std::cout<<"ERROR! the dimensions of the gravito inertial wrench and of the HS description are NOT consistent"<<std::endl;
		std::cout<<"the wrench has "<<wrench_gi.rows()<<" terms"<<std::endl;
		std::cout<<"the halfspaces have "<<M1.cols()<<" columns"<<std::endl;
	}

//	std::cout<<"point: "<<p<<std::endl;
//	std::cout<<"HS descr: "<<M_tmp<<std::endl;

//	remove_hs_with_null_norm(M_tmp, M);
	std::cout<<"non redundant HS descr: "<<M_tmp<<std::endl;
	M = M_tmp.block(0,0,M1.rows(),M1.cols());
	normalize_half_spaces(M);
	std::cout<<"normalized halfspaces: "<<M<<std::endl;

	residual_radius_LP(M_tmp, wrench_gi, cws_feasibility);
	std::cout<<"compute distance point to hyperplane"<<std::endl;

	// This equation assumes that the hyperplanes pass through the origin!
	// The know term b is in the first column of the matrix M
//	for (int i=0;i<M.rows();i++){
//		//			std::cout<<*(A->matrix[i][0])<<" "<<*(A->matrix[i][1])<<" "<<*(A->matrix[i][2])<<" "<<*(A->matrix[i][3])<<" "<<*(A->matrix[i][4])<<" "<<*(A->matrix[i][5])<<std::endl;
//		denum = pow(M(i,0),2) + pow(M(i,1),2) +
//				pow(M(i,2),2) + pow(M(i,3),2) +
//				pow(M(i,4),2) + pow(M(i,5),2);
//		denum = sqrt(denum);
//		num =   M(i,0)*p(rbd::AX) + M(i,1)*p(rbd::AY) +
//				M(i,2)*p(rbd::AZ) + M(i,3)*p(rbd::LX) +
//				M(i,4)*p(rbd::LY) + M(i,5)*p(rbd::LZ);
//		dist = num/denum;
//
//		if(dist < margin) margin = dist;
//
//	}
//
//	//		std::cout<<"feasibility margin "<< margin << std::endl;
//	if (margin<5*10e3){
//		cws_feasibility = margin;
//		//		old_cws_feasibility = -1;
//	}else{
//		cws_feasibility = -10;
//		std::cout<<"cws feasibility margin is inconsistent"<<std::endl;
//	}
}

rbd::ForceVector FeasibleWrenchPolytope_API::getGravitoInertialWrench()
{
    return inertial_wrench_WF - grav_wrench_WF;
}

	//  Here we compute the rotation matrices that rotate the surface normal vector of an
	//  angle corresponding to half of the pyramid vertex angle in the four possible combinations
void FeasibleWrenchPolytope_API::NormalToEdgesRotation(const double half_cone_angle,
		const unsigned int edges_per_cone,
		iit::dog::LegDataMap<Eigen::Matrix3d> & R){

	iit::dog::LegDataMap<Eigen::Matrix3d> Rx, Ry;
	Eigen::Matrix3d R_tmp;
	Eigen::Matrix3d R_init = RigidBodyDynamics::Math::roty(half_cone_angle);
	Eigen::Matrix3d R_tmp2 = RigidBodyDynamics::Math::rotz(45.0/180.0*3.1415);
	for (int i = 0; i<4;i++){
		R[i].setZero();
	}

	double angle;
	for (int i = 0; i<edges_per_cone;i++){
		angle = 2*M_PI*(double)i/(double)edges_per_cone;
//		std::cout<<angle<<std::endl;
		R_tmp = RigidBodyDynamics::Math::rotz(angle);
		R[i] = R_tmp2.transpose()*R_tmp.transpose()*R_init;
	}

}

double FeasibleWrenchPolytope_API::GetVectorLenght(Eigen::Vector3d vec){
	double tmp = pow(vec(rbd::X),2) + pow(vec(rbd::Y),2)+ pow(vec(rbd::Z),2);
	return tmp = sqrt(tmp);

}


//void FeasibleWrenchPolytope_API::eigen2politopix(const iit::dog::LegDataMap<rbd::ForceVector> edge, boost::shared_ptr<Polytope_Rn> & polytope){
////
//	boost::shared_ptr<Generator_Rn> VX0(new Generator_Rn(cardinality));
//	boost::shared_ptr<Generator_Rn> VX1(new Generator_Rn(cardinality));
//	boost::shared_ptr<Generator_Rn> VX2(new Generator_Rn(cardinality));
//	boost::shared_ptr<Generator_Rn> VX3(new Generator_Rn(cardinality));
//	boost::shared_ptr<Generator_Rn> VX4(new Generator_Rn(cardinality));
//	boost::shared_ptr<Generator_Rn> VX;
//
//	polytope.reset(new Polytope_Rn());
//	//		fill in all the edges
//
//	unsigned int ax = 0, ay = 1, az = 2, lx = 3, ly = 4, lz = 5;
//	VX0->setCoordinate(ax, (double)0.0);
//	VX0->setCoordinate(ay, (double)0.0);
//	VX0->setCoordinate(az, (double)0.0);
//	VX0->setCoordinate(lx, (double)0.0);
//	VX0->setCoordinate(ly, (double)0.0);
//	VX0->setCoordinate(lz, (double)0.0);
//	polytope->addGenerator(VX0);
//
//	VX1->setCoordinate(ax, (double)edge[0](rbd::AX));
//	VX1->setCoordinate(ay, (double)edge[0](rbd::AY));
//	VX1->setCoordinate(az, (double)edge[0](rbd::AZ));
//	VX1->setCoordinate(lx, (double)edge[0](rbd::LX));
//	VX1->setCoordinate(ly, (double)edge[0](rbd::LY));
//	VX1->setCoordinate(lz, (double)edge[0](rbd::LZ));
//	polytope->addGenerator(VX1);
//
//	VX2->setCoordinate(ax, (double)edge[1](rbd::AX));
//	VX2->setCoordinate(ay, (double)edge[1](rbd::AY));
//	VX2->setCoordinate(az, (double)edge[1](rbd::AZ));
//	VX2->setCoordinate(lx, (double)edge[1](rbd::LX));
//	VX2->setCoordinate(ly, (double)edge[1](rbd::LY));
//	VX2->setCoordinate(lz, (double)edge[1](rbd::LZ));
//	polytope->addGenerator(VX2);
//
//	VX3->setCoordinate(ax, (double)edge[2](rbd::AX));
//	VX3->setCoordinate(ay, (double)edge[2](rbd::AY));
//	VX3->setCoordinate(az, (double)edge[2](rbd::AZ));
//	VX3->setCoordinate(lx, (double)edge[2](rbd::LX));
//	VX3->setCoordinate(ly, (double)edge[2](rbd::LY));
//	VX3->setCoordinate(lz, (double)edge[2](rbd::LZ));
//	polytope->addGenerator(VX3);
//
//	VX4->setCoordinate(ax, (double)edge[3](rbd::AX));
//	VX4->setCoordinate(ay, (double)edge[3](rbd::AY));
//	VX4->setCoordinate(az, (double)edge[3](rbd::AZ));
//	VX4->setCoordinate(lx, (double)edge[3](rbd::LX));
//	VX4->setCoordinate(ly, (double)edge[3](rbd::LY));
//	VX4->setCoordinate(lz, (double)edge[3](rbd::LZ));
//	polytope->addGenerator(VX4);
//
//}

//ddf_MatrixPtr FeasibleWrenchPolytope_API::eigen2cdd(iit::dog::LegDataMap<rbd::ForceVector> edge){
////	  ddf_set_global_constants();
//	  ddf_NumberType numb = ddf_Real;   /* set a number type */
//	  ddf_rowrange m=4;    /* number of rows  */
//	  ddf_colrange n=7;    /* number of columns */
//	  ddf_MatrixPtr A=ddf_CreateMatrix(m,n);
//
//	  double randnum;
//	  signed long edgeval;
//	  srand((int) time(0));
//	  for(int i=0; i<m; i++){
//
////		  min + rand() % (max - min);
//		  ddf_set_si(A->matrix[i][0], 0);
//
//		  randnum = (rand() % 50) - 25;
//		  edgeval = edge[i](rbd::AX) + randnum;
////		  std::cout<<edgeval<<std::endl;
//		  ddf_set_si(A->matrix[i][1], edgeval);
//
//          randnum = (rand() % 50) - 25;
//		  edgeval = edge[i](rbd::AY) + randnum;
////		  std::cout<<edgeval<<std::endl;
//		  ddf_set_si(A->matrix[i][2], edgeval);
//
//          randnum = (rand() % 50) - 25;
//		  edgeval = edge[i](rbd::AZ);
////		  std::cout<<edgeval<<std::endl;
//		  ddf_set_si(A->matrix[i][3], edgeval);
//
//          randnum = (rand() % 50) - 25;
//		  edgeval = edge[i](rbd::LX) + randnum;
////		  std::cout<<edgeval<<std::endl;
//		  ddf_set_si(A->matrix[i][4],  edgeval);
//
//          randnum = (rand() % 50) - 25;
//		  edgeval = edge[i](rbd::LY) + randnum;
////		  std::cout<<edgeval<<std::endl;
//		  ddf_set_si(A->matrix[i][5], edgeval);
//
//          randnum = (rand() % 50) - 25;
//		  edgeval = edge[i](rbd::LZ) + randnum;
////		  std::cout<<edgeval<<std::endl;
//		  ddf_set_si(A->matrix[i][6], edgeval);
//	  }
//	  return A;
//}
//
//
//void FeasibleWrenchPolytope_API::cdd2eigen(ddf_MatrixPtr A, Eigen::MatrixXd & A_eig){
//	int n = A->colsize;
//	int m = A->rowsize;
//	A_eig.resize(m,n);
//	for (int j=0; j<m; j++){
//		for(int i=0;i<n; i++){
//			A_eig(j,i) = *(A->matrix[j][i]);
//		}
//	}
//
//}

//void FeasibleWrenchPolytope_API::cdd2politopix(ddf_MatrixPtr A, boost::shared_ptr<Polytope_Rn> & poly){
//	int n = A->colsize;
//	int m = A->rowsize;
//	poly.reset(new Polytope_Rn());
//	unsigned int ax = 0, ay = 1, az = 2, lx = 3, ly = 4, lz = 5;
//
//	Rn::setDimension(cardinality);
//	for (int j=0; j<m; j++){
//		boost::shared_ptr<Generator_Rn> VX(new Generator_Rn(cardinality));
//		VX->setCoordinate(ax, *(A->matrix[j][ax]));
//		VX->setCoordinate(ay, *(A->matrix[j][ay]));
//		VX->setCoordinate(az, *(A->matrix[j][az]));
//		VX->setCoordinate(lx, *(A->matrix[j][lx]));
//		VX->setCoordinate(ly, *(A->matrix[j][ly]));
//		VX->setCoordinate(lz, *(A->matrix[j][lz]));
//		poly->addGenerator(VX);
//        VX.reset(new Generator_Rn(cardinality));
//	}
//
//}


//void FeasibleWrenchPolytope_API::getPointsNum(unsigned int & pointsNumber){
//
//	pointsNumber = cardinality;
//}
//
////void FeasibleWrenchPolytope_API::debugPolytope(boost::shared_ptr<Polytope_Rn> polytope){
////    polytope->checkFacets();
////    polytope->checkEdges();
////
////    int gen_size = polytope->numberOfGenerators();
////
//////    for(unsigned int i=1; i<=gen_size; i++){
//////        for(unsigned int j=1; j<=gen_size; j++){
//////            polytope->checkDuplicateGenerators(i,j);
//////        }
//////    }
//
//    std::cout<< "num of gens: "<<gen_size <<std::endl;
//    listOfGeometricObjects< boost::shared_ptr<Generator_Rn> > listOfEdges = polytope->getListOfGenerators();
//    constIteratorOfListOfGeometricObjects< boost::shared_ptr<Generator_Rn> > gen_iterator(listOfEdges);
//    for (gen_iterator.begin(); gen_iterator.end()!=true; gen_iterator.next()) {
//      const boost::shared_ptr<Generator_Rn>& currentGenerator = gen_iterator.current();
//      gen_size = currentGenerator->dimension();
//      boost::numeric::ublas::vector<double> GenCoord = currentGenerator->vect();
//      std::cout<< "gen size: "<<gen_size<<" coord:"<<GenCoord <<std::endl;
//    }
//
//    double hs_size;
//    constIteratorOfListOfGeometricObjects< boost::shared_ptr<HalfSpace_Rn> > hs_iterator(polytope->getListOfHalfSpaces());
//    for (hs_iterator.begin(); hs_iterator.end()!=true; hs_iterator.next()) {
//      const boost::shared_ptr<HalfSpace_Rn>& currentHalfSpace = hs_iterator.current();
//      hs_size = currentHalfSpace->dimension();
//      boost::numeric::ublas::vector<double> HScoord = currentHalfSpace->vect();
//      std::cout<<"coord:"<<HScoord <<std::endl;
//    }
//
//}


void FeasibleWrenchPolytope_API::CreateEdges(const iit::dog::LegDataMap<rbd::Vector3d> stance_feet_WF,
								const iit::dog::LegDataMap<double> friction_coeff,
								const iit::dog::LegDataMap<rbd::Vector3d> terrain_normal,
								const iit::dog::LegDataMap<double> max_normal_grf,
								iit::dog::LegDataMap< Eigen::MatrixXd > & legs_set){

	iit::dog::LegDataMap< Eigen::MatrixXd > legs_friction_cone;
	legs_friction_cone_3d(friction_coeff,
							terrain_normal,
							max_normal_grf,
							legs_friction_cone);

	for (int leg=dog::LF; leg<=dog::RH; leg++)
	{

		fill_angular_component(stance_feet_WF[leg], legs_friction_cone[leg], legs_set[leg]);
		//		for (int edge=0; edge<edges_per_cone; edge++)
//		{
//			rbd::linearPart(cws_pyramid[leg][edge]) = friction_cone_edges[leg][edge];
//			rbd::angularPart(cws_pyramid[leg][edge]) = stance_feet_WF[leg].cross(friction_cone_edges[leg][edge]);
//
//			//								std::cout<<"leg: "<< leg<<" edge: "<<edge<<" friction edges: " <<cws_pyramid[leg][edge](rbd::AX)<<
//			//										" "<<cws_pyramid[leg][edge](rbd::AY)<<" "<<cws_pyramid[leg][edge](rbd::AZ)<<
//			//										" "<<cws_pyramid[leg][edge](rbd::LX)<<" "<<cws_pyramid[leg][edge](rbd::LY)<<" "<<cws_pyramid[leg][edge](rbd::LZ)<<std::endl;
//			//			}
//			legs_set[leg].block(0,edge,6,1) = cws_pyramid[leg][edge];
//		}
		// add the fifth vertex of the pyramid (i.e. the origin)
//		legs_set[leg].block(0,legs_vx_num-1,6,1).setZero();
	}
	if(print_all) std::cout<<legs_set<<std::endl;
}

void FeasibleWrenchPolytope_API::torque_limits_wrench_set(const TLSData tlws_struct,
		const Eigen::MatrixXd legs_jacobian,
		const Eigen::Vector3d com_orientation,
		Eigen::MatrixXd & TLWS_v_description){
// The convention is that the vertices are positioned on the columns of the matrix.
// The number of rows of the matrix tells therefore the dimensionality of the problem

	iit::dog::LegDataMap<Eigen::MatrixXd > max_wrench;
	iit::dog::LegDataMap< polytope > max_lin_grf;
//	// fill the linear coordinates of the vertices (last part of the matrix)
	legs_torque_limits_3d(tlws_struct, max_lin_grf);

	// now fill the angular part of the matrix
	for (int leg=dog::LF; leg<=dog::RH; leg++){
		fill_angular_component(tlws_struct.stance_feet_WF[leg], max_lin_grf[leg].v_rep, max_wrench[leg]);
	}

//	std::cout<<torque_max<<std::endl;
//	std::cout<<jacobians<<std::endl;
//	std::cout<<max_wrench<<std::endl;
	MS_LP ms_lp_2;ms_lp_2.setPrintAll(print_all);
	Eigen::MatrixXd sum1;
	Eigen::MatrixXd sum2;
	clock_t start = clock();
	ms_lp_2.mink_sum_with_vertices(max_wrench[dog::LH], max_wrench[dog::LF], TLWS_v_description);
//	ms_lp.mink_sum_with_vertices(M1, M2, A_v_description);
//	std::cout<<"first sum: "<<sum1<<std::endl;
//	ms_lp_2.mink_sum_with_vertices(max_wrench[dog::RF], max_wrench[dog::RH] , sum2);
////	std::cout<<"second sum: "<<sum2<<std::endl;
//	ms_lp_2.mink_sum_with_vertices(sum2, sum1, TLWS_v_description);
//	std::cout<<"third sum: "<<A_v_description<<std::endl;
	clock_t end = clock();


//	check if the the average vertex is in the origin
	Eigen::VectorXd torque_center(6);
	double n_cols = TLWS_v_description.cols();
	double n_rows = TLWS_v_description.rows();
	std::cout<<"n cols "<<n_cols<<"n rows "<<n_rows<<std::endl;
	for (int j=0; j<n_rows; j++){
		torque_center(j) = (TLWS_v_description.block(j,0,1,n_cols)).sum();
	}

	std::cout<<"average torque : "<<torque_center(0)<<" "<<torque_center(1)<<" "<<torque_center(2)<<" "<<torque_center(3)<<" "<<torque_center(4)<<" "<<torque_center(5)<<std::endl;

}

void FeasibleWrenchPolytope_API::joint_space_zonotope(const Eigen::Vector3d joints_lim, Eigen::Matrix<double, 3, 8> & joint_space_zono){
	Eigen::Matrix<double, 8, 3> combos;

	combos << 1.0,   1.0,  1.0,
			 -1.0,   1.0,  1.0,
			  1.0,  -1.0,  1.0,
			 -1.0,  -1.0,  1.0,
			  1.0,   1.0, -1.0,
			 -1.0,   1.0, -1.0,
			  1.0,  -1.0, -1.0,
			 -1.0,  -1.0, -1.0;

	combos.block(0,0,8,1) *= joints_lim(0);
	combos.block(0,1,8,1) *= joints_lim(1);
	combos.block(0,2,8,1) *= joints_lim(2);

	joint_space_zono = combos.transpose();
//	std::cout<<joint_space_zonotope<<std::endl;


}

void FeasibleWrenchPolytope_API::kill(){

}

void FeasibleWrenchPolytope_API::normalize_half_spaces(Eigen::MatrixXd & A_hs){

	unsigned int nRows = A_hs.rows();
	unsigned int nCols = A_hs.cols();

	for (int i=0; i<nRows; i++){
		double norm_row = (A_hs.block(i,0,1,nCols)).norm();
		A_hs.block(i,0,1,nCols) /= norm_row;
	}
}

void FeasibleWrenchPolytope_API::remove_repeated_vertices(boost::shared_ptr<Polytope_Rn> primal_polytope){
	double tolerance = 0.01;
	unsigned int gen_iter1 = 0, gen_iter2 = 0;
	unsigned int redundant;
	bool new_red = false;
	for (gen_iter1 = 0; gen_iter1 < primal_polytope->numberOfGenerators(); gen_iter1++) {
//		std::cout<<"Checking politope "<<gen_iter1<<std::endl;
		// Inner loop compares generator gen_iter1 with each subsequent generator gen_iter2
		for (gen_iter2 = gen_iter1+1; gen_iter2 < primal_polytope->numberOfGenerators(); gen_iter2++) {
//			std::cout<<primal_polytope->getGenerator(gen_iter1)->getCoordinate(0)<<std::endl;
			if(     (fabs(primal_polytope->getGenerator(gen_iter1)->getCoordinate(0) - primal_polytope->getGenerator(gen_iter2)->getCoordinate(0))<= 0.001)&&
					(fabs(primal_polytope->getGenerator(gen_iter1)->getCoordinate(1) - primal_polytope->getGenerator(gen_iter2)->getCoordinate(1))<= 0.001)&&
					(fabs(primal_polytope->getGenerator(gen_iter1)->getCoordinate(2) - primal_polytope->getGenerator(gen_iter2)->getCoordinate(2))<= 0.001)&&
					(fabs(primal_polytope->getGenerator(gen_iter1)->getCoordinate(3) - primal_polytope->getGenerator(gen_iter2)->getCoordinate(3))<= 0.001)&&
					(fabs(primal_polytope->getGenerator(gen_iter1)->getCoordinate(4) - primal_polytope->getGenerator(gen_iter2)->getCoordinate(4))<= 0.001)&&
					(fabs(primal_polytope->getGenerator(gen_iter1)->getCoordinate(5) - primal_polytope->getGenerator(gen_iter2)->getCoordinate(5))<= 0.001)){
				std::cout<<"vertex "<<gen_iter1<<" is redundant"<<std::endl;
				redundant = gen_iter1;
				new_red = true;
			}
		}

		if(new_red){
			primal_polytope->removeGenerator(redundant);
			new_red = false;
		}

	}
}

void FeasibleWrenchPolytope_API::remove_hs_with_null_norm(const Eigen::MatrixXd A_hs, Eigen::MatrixXd & A_hs_reduced){
	double cols = A_hs.cols();
	double rows = A_hs.rows();
	//	std::cout<<"redundant hs description"<<std::endl;
	//	std::cout<<A_hs<<std::endl;
	Eigen::MatrixXd tmp_A_hs;
	unsigned int counter = 0;
	std::vector<double> vec_indx;
	for(int i = 0; i<rows; i++){
		if((A_hs(i,0)<0.1)&&(A_hs(i,0)>-0.1)){
			vec_indx.push_back(i);
		}
	}

	A_hs_reduced.resize(vec_indx.size(), cols);
	A_hs_reduced.setZero();
	for(int k = 0; k< vec_indx.size(); k++){
		A_hs_reduced.block(k,1,1,cols-1) = A_hs.block(vec_indx[k],1,1,cols-1);
		A_hs_reduced(k,0) = 0.0;
	}

}

void FeasibleWrenchPolytope_API::fill_angular_component(const Eigen::Vector3d stance_feet_pos,
										const Eigen::MatrixXd polytope_3d,
										Eigen::MatrixXd & polytope_6d){
	if(polytope_3d.rows() != 3){
		std::cout<< "warning! the input polytope must have 3 rows"<<std::endl;
	}

	unsigned int vertices_n = polytope_3d.cols();
	polytope_6d.resize(6,vertices_n);
	polytope_6d.setZero();
	polytope_6d.block(3,0,3,vertices_n) = polytope_3d;
	for (unsigned int vertex = 0; vertex < vertices_n; vertex++){
		// fill the angular coordinates of the vertices (upper part of the matrix)
		Eigen::Vector3d linear_part = polytope_3d.block(0, vertex, 3, 1);
		Eigen::Matrix3d skew_foot_pos = iit::rbd::Utils::buildCrossProductMatrix(stance_feet_pos);
		polytope_6d.block(0, vertex, 3, 1) = skew_foot_pos*linear_part;
	}
}


void FeasibleWrenchPolytope_API::legs_torque_limits_3d(const FeasibleWrenchPolytope_API::TLSData tls,
										iit::dog::LegDataMap< polytope > & max_lin_grf){


	iit::dog::LegDataMap<Eigen::Matrix<double, 3, 8> > torque_max;
	Eigen::MatrixXd jac_transp;
	iit::dog::LegDataMap<Eigen::Matrix3d> jacobians;
	//TODO do this shit in a proper way
//    std::cout<<tls.fixed_base_jac<<std::endl;
	jacobians[dog::LF] = tls.fixed_base_jac.block(0,0,3,3);
	jacobians[dog::RF] = tls.fixed_base_jac.block(6,6,3,3);
	jacobians[dog::LH] = tls.fixed_base_jac.block(3,3,3,3);
	jacobians[dog::RH] = tls.fixed_base_jac.block(9,9,3,3);
	if (print_all){
        std::cout<<"floating base jacobian:"<<std::endl;
        std::cout<<tls.fixed_base_jac<<std::endl;
        std::cout<<"single leg jacobians:"<<std::endl;
        std::cout<<jacobians<<std::endl;
        std::cout<<"torque limits in the joint space:"<<std::endl;
        std::cout<<tls.max_torque<<std::endl;
        std::cout<<"torques due to the gravity in the legs:"<<std::endl;
        std::cout<<tls.legs_grav_torque<<std::endl;
	}
	iit::dog::LegDataMap<Eigen::Vector3d> tau_lim;
	tau_lim[dog::LF] = tls.max_torque[dog::LF] - tls.legs_grav_torque[dog::LF];
	tau_lim[dog::RF] = tls.max_torque[dog::RF] - tls.legs_grav_torque[dog::RF];
	tau_lim[dog::LH] = tls.max_torque[dog::LH] - tls.legs_grav_torque[dog::LH];
	tau_lim[dog::RH] = tls.max_torque[dog::RH] - tls.legs_grav_torque[dog::RH];

	// fill the linear coordinates of the vertices (last part of the matrix)
	for (int leg=dog::LF; leg<=dog::RH; leg++){
		joint_space_zonotope(tau_lim[leg], torque_max[leg]);
		max_lin_grf[leg].v_rep.resize(3,8);
		max_lin_grf[leg].v_rep.setZero();
		jac_transp = (jacobians[leg]).transpose();
		if(use_simplified_torque_limits_set){
			max_lin_grf[leg].v_rep = torque_max[leg];
		}else{
			max_lin_grf[leg].v_rep = jac_transp.inverse() * torque_max[leg];
			if (print_all) std::cout<<jac_transp.inverse()<<std::endl;
		}

		force_polygon_analytic_hs_rep(max_lin_grf[leg].v_rep, max_lin_grf[leg].hs_rep);
		this->force_polytopes[leg].hs_rep = max_lin_grf[leg].hs_rep;
	}

}

void FeasibleWrenchPolytope_API::legs_friction_cone_3d(const iit::dog::LegDataMap<double> friction_coeff,
		const iit::dog::LegDataMap<rbd::Vector3d> terrain_normal,
		const iit::dog::LegDataMap<double> max_normal_grf,
		iit::dog::LegDataMap< Eigen::MatrixXd > & legs_cone_3d){

	/*
	 * initialize the cws to zero and the half cone angle (angle included between
	 * the terrain normal and the side of the pyramid)
	 */
	for (int leg=dog::LF; leg<=dog::RH; leg++)
	{
		cws_pyramid[leg] = RigidBodyDynamics::Math::SpatialVector::Zero();
		half_cone_angle[leg] = atan(friction_coeff[leg]);
	}
	//std::cout<<half_cone_angle<<std::endl;
	/*
	 * Compute the 4 edges of the friction pyramids for each leg
	 */
	unsigned int edges_per_cone = 4;
	for (int leg=dog::LF; leg<=dog::RH; leg++)
	{
		NormalToEdgesRotation(half_cone_angle[leg], edges_per_cone, R);

		for (int edge=0; edge<edges_per_cone; edge++)
		{
			friction_cone_edges[leg][edge] = R[edge]*terrain_normal[leg];
			//			if(print_friction_edges){
//							std::cout<<"leg: "<< leg<<" edge: "<<edge<<" friction edges: " <<friction_cone_edges[leg][edge](rbd::X)<<" "<<friction_cone_edges[leg][edge](rbd::Y)<<" "<<friction_cone_edges[leg][edge](rbd::Z)<<std::endl;
			//			}
		}
	}

	/*  Now scale the edges to obtain the vertices (in such a way that the normal
	 * component has the maximum value that can be applied to the ground)
	 */
	double projection;
	//	Loop over all the legs
	for (int leg=dog::LF; leg<=dog::RH; leg++)
	{
		legs_cone_3d[leg].resize(3,5);
		legs_cone_3d[leg].setZero();
		//		Loop over all the edges
		for (int edge=0; edge<edges_per_cone; edge++)
		{
			//			double length = GetVectorLenght(friction_cone_edges[leg][edge]);
			projection = cos(half_cone_angle[leg]);
			friction_cone_edges[leg][edge] = friction_cone_edges[leg][edge]/projection*max_normal_grf[leg];
//			friction_cone_edges[leg][edge](rbd::X) = round(friction_cone_edges[leg][edge](rbd::X));
//			friction_cone_edges[leg][edge](rbd::Y) = round(friction_cone_edges[leg][edge](rbd::Y));
//			friction_cone_edges[leg][edge](rbd::Z) = round(friction_cone_edges[leg][edge](rbd::Z));
			legs_cone_3d[leg].block(0,edge,3,1) = friction_cone_edges[leg][edge];
			if(print_all) std::cout<<"leg: "<< leg<<" edge: "<<edge<<" friction edges: " <<friction_cone_edges[leg][edge](rbd::X)<<" "<<friction_cone_edges[leg][edge](rbd::Y)<<" "<<friction_cone_edges[leg][edge](rbd::Z)<<std::endl;
			//			}
		}
	}


}

bool FeasibleWrenchPolytope_API::feasibility_wrench_set(const TLSData tlws_struct,
														const CWSData cws_struct,
														const FWSData fwp_options,
														Eigen::MatrixXd & FWS_v_description){

	iit::dog::LegDataMap<Eigen::MatrixXd > intersected_sets_3d, intersected_sets_6d;

	bool flag = false;
	flag = bounded_friction_polytopes(tlws_struct, cws_struct, fwp_options, intersected_sets_3d);

	for (int leg=dog::LF; leg<=dog::RH; leg++){
		if(fwp_options.wrench_type == full_6D){
			fill_angular_component(cws_struct.stance_feet_WF[leg],
					intersected_sets_3d[leg],
					intersected_sets_6d[leg]);
//			std::cout<<"6d intersection"<<std::endl;
//			std::cout<<intersected_sets_6d[leg]<<std::endl;
		}else{
			//			In this case the intersected_sets_6d is not full: three rows only
			fill_angular_component(cws_struct.stance_feet_WF[leg],
					intersected_sets_3d[leg],
					intersected_sets_3d[leg]);
			(intersected_sets_6d[leg]).resize(3,(intersected_sets_3d[leg]).cols());
			(intersected_sets_6d[leg]).setZero();
			if(fwp_options.wrench_type == angular_3D){// angular only case
				intersected_sets_6d[leg] = intersected_sets_3d[leg].block(0,0,3,(intersected_sets_3d[leg]).cols());
			}else{
				if (fwp_options.wrench_type == linear_3D){// linear only case
					intersected_sets_6d[leg] = intersected_sets_3d[leg].block(3,0,3,(intersected_sets_3d[leg]).cols());
				}
			}
			std::cout<<"3D intersection: "<<(intersected_sets_6d[leg]).transpose()<<std::endl;
		}

		if(print_all) {
			std::cout<<"foot pos"<<std::endl;
			std::cout<<cws_struct.stance_feet_WF[leg]<<std::endl;
			std::cout<<"3d intersection"<<std::endl;
			std::cout<<intersected_sets_3d[leg]<<std::endl;
		}
	}
//	 this only works if we know that there are three stance legs
	Eigen::VectorXi stance_legs;
	stance_legs.resize(4);
	stance_legs.setZero();
	int stance_count = 0;
	for (int leg=dog::LF; leg<=dog::RH; leg++){
		if(cws_struct.stance_legs_flag[leg]) {
			stance_legs(stance_count) = leg;
			if (print_all) std::cout<<"foothold in the optimization: "<<cws_struct.stance_feet_WF[stance_legs(stance_count)]<<std::endl;
			stance_count++;
			if (print_all) std::cout<<"stance_count "<< stance_count <<std::endl;
		}
	}

	MS_LP ms_lp_3;
	ms_lp_3.setPrintAll(print_all);
	Eigen::MatrixXd sum1, sum2, sum3;
	std::cout<<"predicted stance legs: "<<stance_legs.transpose()<<std::endl;

	if(stance_count==3){
		ms_lp_3.mink_sum_with_vertices(intersected_sets_6d[stance_legs(0)],
				intersected_sets_6d[stance_legs(1)],
				sum1);

//		std::cout<<"predicted stance legs: "<<intersected_sets_6d[stance_legs(0)]<<std::endl;
//		std::cout<<"predicted stance legs: "<<intersected_sets_6d[stance_legs(1)]<<std::endl;
//		std::cout<<"predicted stance legs: "<<intersected_sets_6d[stance_legs(2)]<<std::endl;

		ms_lp_3.mink_sum_with_vertices(intersected_sets_6d[stance_legs(2)],
				sum1,
				sum2);

		FWS_v_description.resize(sum2.rows(), sum2.cols());
		FWS_v_description = sum2;
		fwp.v_rep = FWS_v_description;
	}else if(stance_count==4){
		std::cout<<"Four stance legs"<<std::endl;
		ms_lp_3.mink_sum_with_vertices(intersected_sets_6d[stance_legs(0)],
				intersected_sets_6d[stance_legs(1)],
				sum1);

		ms_lp_3.mink_sum_with_vertices(intersected_sets_6d[stance_legs(2)],
				intersected_sets_6d[stance_legs(3)],
				sum2);

		ms_lp_3.mink_sum_with_vertices(sum1,
				sum2,
				sum3);
		FWS_v_description.resize(sum3.rows(), sum3.cols());
		FWS_v_description = sum3;
		fwp.v_rep = FWS_v_description;
	} else {
		std::cout<<"WTF?!?"<<std::endl;
	}

	return flag;

}

void FeasibleWrenchPolytope_API::residual_radius_LP(const hs_description A_hs, const Eigen::VectorXd wrench_gi, double & residual_radius){
//void FeasibleWrenchPolytope_API::residual_radius_LP(){
    std::cout<<"Started computing the residual radius!"<<std::endl;
	clock_t start = clock();
    cheb_center.setInitialState(A_hs, wrench_gi);

	// LP solver
    solver->setOptimizationModel(& cheb_center);
    std::string pack_path = ros::package::getPath(std::string("contact_wrench_set"))+ "/config/ipopt_configLP.yaml";
    solver->setFromConfigFile(pack_path);//should be in the folder
    solver->init();
    if(solver->compute(10.0)){ // max time in seconds
    	Eigen::VectorXd solution = solver->getSolution();
    	residual_radius = solution(0);
    }
    clock_t end = clock();
    float seconds = (float)(end - start) / CLOCKS_PER_SEC;
    printf ("seconds needed by residual radius computation: %f  \n", seconds);

    std::cout<<"Finished computing the residual radius!"<<std::endl;
    std::cout<<"The residual radius is: "<<residual_radius<<std::endl;
}

void FeasibleWrenchPolytope_API::vertex_based_margin_LP(const wrenchType & wrench_type,
														const v_description & A_v, 
														const Eigen::VectorXd & wrench_gi, 
														double & fwp_margin){
    clock_t start = clock();
    //std::cout<<A_v<<std::endl;
    //std::cout<<wrench_gi<<std::endl;
    switch(wrench_type){
    	case full_6D:
    		opt_weights.setInitialState(A_v, wrench_gi);
    		break;
    	case linear_3D:
			opt_weights.setInitialState(A_v, rbd::linearPart(wrench_gi));
			break;
    	case angular_3D:
			opt_weights.setInitialState(A_v, rbd::angularPart(wrench_gi));
    		break;
    }   
    std::cout<<"Set optimization model"<<std::endl;
    margin_solver->setOptimizationModel(& opt_weights);
    std::cout<<"Set path"<<std::endl;
    std::string pack_path = ros::package::getPath(std::string("contact_wrench_set"))+ "/config/ipopt_configLP.yaml";
    std::cout<<"Set from config file"<<std::endl;
    margin_solver->setFromConfigFile(pack_path);//should be in the folder
    std::cout<<"Init the solver"<<std::endl;
    margin_solver->init();
    if(margin_solver->compute(1.0)){ //5.0 max time
    	Eigen::VectorXd solution = margin_solver->getSolution();
    	fwp_margin = solution(A_v.cols());
  	    std::cout<<"The solution is: "<<solution<<std::endl;
  	    //opt_weights.evaluateSolution(solution);
   	    //std::cout<<"The solution is: "<<solution<<std::endl;
    	Eigen::VectorXd weights = solution.segment(0,A_v.cols());
    }
    clock_t end = clock();
    float seconds = (float)(end - start) / CLOCKS_PER_SEC;
    printf ("seconds needed by FWS construction + optimization: %f  \n", seconds);

    std::cout<<"Finished computing the feasibility: v-margin ready!"<<std::endl;
    std::cout<<"The v-margin is: "<<fwp_margin<<std::endl;
}


void FeasibleWrenchPolytope_API::get_jacobian(const int constraint_type,
								const dog::LegDataMap<Eigen::Vector3d> footPos_BF,
								Eigen::MatrixXd & fixed_base_jacobian){
	if((constraint_type == 3)||(constraint_type == 2)){
		std::vector<unsigned int> foot_id_map_;
//		// Resetting the system from the hyq urdf file
		model_ = ros::package::getPath(std::string("contact_wrench_set")) + "/config/hyq.urdf";
		robot_ = ros::package::getPath(std::string("contact_wrench_set")) + "/config/hyq.yarf";
		fbs.resetFromURDFFile(model_, robot_);
		wb_kin.modelFromURDFFile(model_, robot_);
		dog::LegDataMap<Eigen::Vector3d> footPosB;
		for (int leg = iit::dog::LF; leg <= iit::dog::RH; leg++){
			//        	footPosB[leg] = gl.R*(footPosW[leg] - awp_margin.comAWPApproxWF);
			footPosB = footPos_BF;
		}
		dwl::rbd::BodyVector3d ik_pos;
        dwl::rbd::Vector6d base_pos;
        base_pos.setZero();
		dwl::rbd::BodySelector feet_names = fbs.getEndEffectorNames(dwl::model::FOOT);
		foot_id_map_.resize(fbs.getNumberOfEndEffectors(dwl::model::FOOT));
		foot_id_map_[dog::LF] = fbs.getEndEffectorId("lf_foot");
		foot_id_map_[dog::RF] = fbs.getEndEffectorId("rf_foot");
		foot_id_map_[dog::LH] = fbs.getEndEffectorId("lh_foot");
		foot_id_map_[dog::RH] = fbs.getEndEffectorId("rh_foot");

		for (int foot = dog::LF; foot <= dog::RH; foot++) {
			int id = foot_id_map_[foot];
			std::string name = fbs.getEndEffectorNames()[id];
			ik_pos[name] = footPosB[foot];
//			std::cout<<ik_pos[name]<<std::endl;
		}

        wb_kin.computeInverseKinematics(base_pos, current_wbs.joint_pos, ik_pos);

		Eigen::MatrixXd full_base_jacobian;
		wb_kin.computeJacobian(full_base_jacobian,
				current_wbs.base_pos, current_wbs.joint_pos,
				fbs.getEndEffectorNames(), dwl::rbd::Linear);
//		std::cout<<full_base_jacobian<<std::endl;

		wb_kin.getFixedBaseJacobian(fixed_base_jacobian, full_base_jacobian);
	}else{
		fixed_base_jacobian.resize(1,1);
		fixed_base_jacobian.setZero();
	}

}

bool FeasibleWrenchPolytope_API::compute_intersection(FeasibleWrenchPolytope_API::v_description friction_cone,
										FeasibleWrenchPolytope_API::v_description force_polytope,
										const int leg_id,
										FeasibleWrenchPolytope_API::v_description & intersection_3d){

	unsigned int dim = friction_cone.rows(); // dimensionality of each vertex
//	unsigned int vx_num = friction_cone_v.cols(); // num of vertices
	double margin_tol = 10e-7;
	bool flag;
	Rn::setDimension(dim);
	Rn::setTolerance(margin_tol);
	boost::shared_ptr<Polytope_Rn> cws_3d_polytope(new Polytope_Rn());
	boost::shared_ptr<Polytope_Rn> tlws_3d_polytope(new Polytope_Rn());
	boost::shared_ptr<Polytope_Rn> Intersection(new Polytope_Rn());

	if(print_all) std::cout<<"v fr: "<<friction_cone<<std::endl;
	if(print_all) std::cout<<"v tl: "<<force_polytope<<std::endl;
	get_hs_description_politopix(force_polytope, tlws_3d_polytope);

	unsigned int fp_hs_num = tlws_3d_polytope->numberOfHalfSpaces();
	std::cout<<"v tl: "<<fp_hs_num<<std::endl;
	//	if(fp_hs_num==6){
	build_topology(tlws_3d_polytope, fp_top_map, fp_top_vec);
	clock_wise_sort(force_polytope, fp_top_vec, fp_top_map);
	force_polytopes[leg_id].v_rep.resize(force_polytope.rows(), force_polytope.cols());
	force_polytopes[leg_id].v_rep = force_polytope;

	get_hs_description_politopix(friction_cone, cws_3d_polytope);
	build_topology(cws_3d_polytope, lfc_top_map, lfc_top_vec);
//	check_friction_cone_topology(cws_3d_polytope, lfc_top_map, lfc_top_vec);
	//		std::cout<<"before swapping: "<<lfc_top_map<<std::endl;
	clock_wise_sort(friction_cone, lfc_top_vec, lfc_top_map);
	linear_friction_cones[leg_id].resize(friction_cone.rows(), friction_cone.cols());
	linear_friction_cones[leg_id] = friction_cone;
	//		std::cout<<"after swapping: "<<lfc_top_map<<std::endl;

	if(print_all) std::cout<<"compute intersection"<<std::endl;

	politopixAPI::computeIntersection(cws_3d_polytope,
			tlws_3d_polytope,
			Intersection);

	eigen_v_description(Intersection, intersection_3d);
	reorder_v_description(Intersection, intersection_3d);
	build_topology(Intersection, bfc_top_map, bfc_top_vec);
	std::cout<<bfc_top_vec<<std::endl;
	clock_wise_sort(intersection_3d, bfc_top_vec, bfc_top_map);
	flag = true;
	//	}else{
		//		intersection_3d = bounded_friction_poly[leg_id];
	//		flag = false;
	//	}

	cws_3d_polytope.reset(new Polytope_Rn());
	tlws_3d_polytope.reset(new Polytope_Rn());
	Intersection.reset(new Polytope_Rn());
	return flag;

}

bool FeasibleWrenchPolytope_API::bounded_friction_polytopes(const TLSData tlws_struct,
		const CWSData cws_struct,
		const FWSData fwp_options,
		iit::dog::LegDataMap<Eigen::MatrixXd > & bounded_fp){

//	Eigen::MatrixXd legs_jacobian;
//	legs_jacobian = tlws_struct.fixed_base_jac;
	bool flag = false;
	iit::dog::LegDataMap<bool> flags_list; flags_list = false;

	// The convention is that the vertices are positioned on the columns of the matrix.
	// The number of rows of the matrix tells therefore the dimensionality of the problem
	std::cout<<"Entered FWS method"<<std::endl;
	if(print_all) std::cout<<"jacobian is: "<<tlws_struct.fixed_base_jac<<std::endl;
	iit::dog::LegDataMap<Eigen::MatrixXd > friction_cone_v;
	iit::dog::LegDataMap< polytope > max_lin_grf_v;

	legs_friction_cone_3d(cws_struct.friction,
			cws_struct.normal,
			cws_struct.max_normal_force,
			friction_cone_v);

	for (int leg=dog::LF; leg<=dog::RH; leg++){
		if (print_all) std::cout<<"=========================> leg number: "<<leg<<" <=============================================="<<std::endl;
		switch(fwp_options.constraint_type){
		case only_friction:
			if (print_all) std::cout<<"only friction constraints considered."<<std::endl;
			bounded_fp[leg] = friction_cone_v[leg];
			flags_list[leg] = true;
			linear_friction_cones[leg] = bounded_fp[leg];
			bounded_friction_poly[leg].v_rep = bounded_fp[leg];
			if (print_all) std::cout<<"3D friction cone: "<<(bounded_fp[leg]).transpose()<<std::endl;
			break;
		case only_actuation:
			if (print_all) std::cout<<" only actuation constraints considered."<<std::endl;
			legs_torque_limits_3d(tlws_struct, max_lin_grf_v);
			bounded_fp[leg] = max_lin_grf_v[leg].v_rep;
			flags_list[leg] = true;
			force_polytopes[leg].v_rep = bounded_fp[leg];
			bounded_friction_poly[leg].v_rep = bounded_fp[leg];
			if (print_all) std::cout<<"force polytope cone: "<<(bounded_fp[leg]).transpose()<<std::endl;
			break;
		case friction_and_actuation:
			if (print_all) std::cout<<" both actuation and friction constraints considered"<<std::endl;
			// fill the linear coordinates of the vertices (last part of the matrix)
			legs_torque_limits_3d(tlws_struct, max_lin_grf_v);
			std::cout<<"Compute intersection"<<std::endl;
			flags_list[leg] = compute_intersection(friction_cone_v[leg], max_lin_grf_v[leg].v_rep, leg, bounded_fp[leg]);
			if(flags_list[leg]) bounded_friction_poly[leg].v_rep = bounded_fp[leg];
			force_polytopes[leg].v_rep = max_lin_grf_v[leg].v_rep;
			linear_friction_cones[leg] = friction_cone_v[leg];
			break;
		}

	}

	flag = flags_list[0]||flags_list[1]||flags_list[2]||flags_list[3];
//	bounded_friction_poly = bounded_fp;
	return flag;

}

void FeasibleWrenchPolytope_API::get_bounded_friction_polytopes(iit::dog::LegDataMap< polytope > & bfp){
	for (int leg = iit::dog::LF; leg <= iit::dog::RH; leg++) {
		(bfp[leg].v_rep).resize(bounded_friction_poly[leg].v_rep.rows(),bounded_friction_poly[leg].v_rep.cols());
		bfp[leg].v_rep = this->bounded_friction_poly[leg].v_rep;
		bfp[leg].top_map = bfc_top_map;
		bfp[leg].top_vec = bfc_top_vec;
//		std::cout<<(bfp[iit::dog::RF]).transpose()<<std::endl;
	}
}

void FeasibleWrenchPolytope_API::get_linear_friction_cones(iit::dog::LegDataMap< polytope > & lfc){
	for (int leg = iit::dog::LF; leg <= iit::dog::RH; leg++) {
		(lfc[leg].v_rep).resize(linear_friction_cones[leg].rows(),linear_friction_cones[leg].cols());
		lfc[leg].v_rep = linear_friction_cones[leg];
		lfc[leg].top_map = lfc_top_map;
		lfc[leg].top_vec = lfc_top_vec;
//		std::cout<<(lfc[iit::dog::RF]).transpose()<<std::endl;
	}
}

void FeasibleWrenchPolytope_API::get_force_polytopes(iit::dog::LegDataMap< polytope> & fp){
	for (int leg = iit::dog::LF; leg <= iit::dog::RH; leg++) {
		(fp[leg].v_rep).resize(force_polytopes[leg].v_rep.rows(),force_polytopes[leg].v_rep.cols());
		fp[leg].v_rep = force_polytopes[leg].v_rep;
		fp[leg].hs_rep = force_polytopes[leg].hs_rep;
		fp[leg].top_map = fp_top_map;
		fp[leg].top_vec = fp_top_vec;
//		std::cout<<(fp[iit::dog::RF]).transpose()<<std::endl;
	}
}

void FeasibleWrenchPolytope_API::get_fwp(polytope & fw_poly){

	int cols = fwp.v_rep.cols();
	int rows = fwp.v_rep.rows();
	fw_poly.v_rep.resize(rows, cols);
	fw_poly.v_rep = fwp.v_rep;
	fw_poly.top_map = fwp_top_map;
	fw_poly.top_vec = fwp_top_vec;

}

void FeasibleWrenchPolytope_API::build_topology(const boost::shared_ptr<Polytope_Rn> primal_polytope,
								topology_map & topology,
								topology_vec & top_vec){
	unsigned int hs_size = primal_polytope->numberOfHalfSpaces();
	unsigned int dim = primal_polytope->dimension();
	topology.resize(hs_size,30);
	topology.setConstant(10000);
	top_vec.resize(hs_size);
	top_vec.setZero();
	bool is_inside;
	constIteratorOfListOfGeometricObjects< boost::shared_ptr<HalfSpace_Rn> > hs_iterator(primal_polytope->getListOfHalfSpaces());
	unsigned int i = 0;
	for (hs_iterator.begin(); hs_iterator.end()!=true; hs_iterator.next()) {
		const boost::shared_ptr<HalfSpace_Rn>& currentHalfSpace = hs_iterator.current();
//		std::cout<< currentHalfSpace->dimension()<<std::endl;
		constIteratorOfListOfGeometricObjects< boost::shared_ptr<Generator_Rn> > v_iterator(primal_polytope->getListOfGenerators());
		unsigned int j = 0;
		for (v_iterator.begin(); v_iterator.end()!=true; v_iterator.next()) {
			const boost::shared_ptr<Generator_Rn>& currentGenerator = v_iterator.current();
			is_inside = currentGenerator->isFacetInside(currentHalfSpace);

			if(is_inside){
				int gen_index = v_iterator.currentIteratorNumber();
				topology(i,j) = gen_index;
				j++;
				is_inside = false;
			}

		}
		top_vec(i) = j;
		i++;
	}



}

void FeasibleWrenchPolytope_API::clock_wise_sort(const FeasibleWrenchPolytope_API::v_description v_rep,
									const FeasibleWrenchPolytope_API::topology_vec top_vec,
									FeasibleWrenchPolytope_API::topology_map & top_map){

	FeasibleWrenchPolytope_API::topology_map new_top_map;
	new_top_map.resize(top_map.rows(),top_map.cols());
	new_top_map.setZero();
	new_top_map = top_map;
//	std::cout<<v_rep<<std::endl;
	int hs_num = top_map.rows();
	Eigen::Vector3d cross_prod, line1, line2;
	for(int hs_iter = 0; hs_iter < hs_num; hs_iter++){
//		std::cout<< "Half space number: "<< hs_iter <<std::endl;
		int v_num = top_vec(hs_iter);
//		std::cout<< "Vertices number of this hs: "<< v_num <<std::endl;
		Eigen::Vector3d average;
		average.setZero();
		for(int v_iter = 0; v_iter < v_num; v_iter++){
			average += v_rep.block(0,top_map(hs_iter, v_iter),3,1);
		}
		average /= v_num;
//		std::cout<<average<<std::endl;

		std::vector<double> angles(v_num);
		Eigen::VectorXd angles_copy;
		angles_copy.resize(v_num);
		angles_copy.setZero();

		for(int v_iter = 0; v_iter < v_num; v_iter++){
			Eigen::Vector3d current_vec = v_rep.block(0, top_map(hs_iter, v_iter),3,1);
//			std::cout<< current_vec.transpose() <<std::endl;
			Eigen::Vector3d current_vec_rot = current_vec - average;//Rot.transpose()*current_vec;
//			std::cout<< current_vec_rot.transpose() <<std::endl;
			angles[v_iter] = atan2(current_vec_rot(1),current_vec_rot(0));
			angles_copy(v_iter) = angles[v_iter];
		}
//		for (std::vector<int>::size_type i = 0; i != angles.size(); ++i)
//			cout << angles[i] << " ";
//
//		cout << endl;
		sort(angles.begin(), angles.end());
//		for (std::vector<int>::size_type i = 0; i != angles.size(); ++i)
//			cout << angles[i] << " ";
//
//		    cout << endl;
		Eigen::VectorXi new_top_map_row;
		new_top_map_row.resize(v_num);
		new_top_map_row.setZero();
		for(int v_iter = 0; v_iter < v_num; v_iter++){
			for(int v_iter2 = 0; v_iter2 < v_num; v_iter2++){

				if (fabs(angles[v_iter] - angles_copy(v_iter2)) < 0.01){
//					std::cout<<top_map(hs_iter, v_iter2)<<std::endl;
					new_top_map(hs_iter, v_iter) = top_map(hs_iter, v_iter2);
				}
			}
		}


	}
	top_map = new_top_map;

}


void FeasibleWrenchPolytope_API::draw_polygon_vertices(iit::dog::LegDataMap< polytope > poly,
								iit::dog::LegDataMap<Eigen::Vector3d > foot_pos,
								dwl::Color color,
								double scaling_factor,
								std::shared_ptr<dwl_rviz_plugin::DisplayInterface> display_){

	for (int leg = iit::dog::RH; leg <= iit::dog::RH; leg++) {
		draw_polygon_vertices(poly[leg], foot_pos[leg], color, scaling_factor, display_);
	}
}

void FeasibleWrenchPolytope_API::draw_polygon_vertices(polytope poly,
					Eigen::Vector3d  application_point,
					dwl::Color color,
					double scaling_factor,
					std::shared_ptr<dwl_rviz_plugin::DisplayInterface> display_){
	dwl::ArrowProperties noArrow(0.02, 0.0, 0.0);
	noArrow.head_length = 0.01;
	//	for (int leg = iit::dog::LF; leg <= iit::dog::RH; leg++) {
	int vert_num_fc = poly.v_rep.cols();

	for (int vertex = 0; vertex<vert_num_fc; vertex++){
		Eigen::Vector3d current_vertex = poly.v_rep.block(0,vertex,3,1) +  application_point;
		current_vertex /= scaling_factor;
		display_->drawSphere(current_vertex + application_point,
				0.05,
				color,
				"world");
	}
	int vert_dim = poly.v_rep.rows();
	if(vert_dim > 3){
		for (int vertex = 0; vertex<vert_num_fc; vertex++){
			Eigen::Vector3d current_vertex = poly.v_rep.block(3,vertex,3,1) +  application_point;
			current_vertex /= scaling_factor;
			display_->drawSphere(current_vertex + application_point,
					0.05,
					dwl::Color(dwl::ColorType::Grey, 0.5),
					"world");
		}
	}

}

void FeasibleWrenchPolytope_API::draw_polygon_edges(iit::dog::LegDataMap< polytope > poly,
		iit::dog::LegDataMap<Eigen::Vector3d > foot_pos,
		dwl::Color color,
		double scaling_factor,
		std::shared_ptr<dwl_rviz_plugin::DisplayInterface> display_){

	dwl::ArrowProperties noArrow(0.02, 0.0, 0.0);
	noArrow.head_length = 0.01;
	for (int leg = iit::dog::RH; leg <= iit::dog::RH; leg++) {
		int hs_num = poly[leg].top_vec.rows();
		for(int hs_iter = 0; hs_iter < hs_num; hs_iter++){
			int v_num = poly[leg].top_vec(hs_iter);
//			std::cout<<poly[leg].top_vec.transpose()<<std::endl;
			for (int v_iter = 0; v_iter < v_num - 1; v_iter++){
				int vert_ind = poly[leg].top_map(hs_iter, v_iter);
				int next_vert_ind = poly[leg].top_map(hs_iter, v_iter+1);
				display_->drawArrow(poly[leg].v_rep.block(0,vert_ind,3,1)/scaling_factor + foot_pos[leg],
						poly[leg].v_rep.block(0,next_vert_ind,3,1)/scaling_factor + foot_pos[leg],
						noArrow,
						color,
						"base_link");
			}

			int vert_ind = poly[leg].top_map(hs_iter, v_num-1);
			int next_vert_ind = poly[leg].top_map(hs_iter, 0);
			display_->drawArrow(poly[leg].v_rep.block(0,vert_ind,3,1)/scaling_factor + foot_pos[leg],
					poly[leg].v_rep.block(0,next_vert_ind,3,1)/scaling_factor + foot_pos[leg],
					noArrow,
					color,
					"base_link");

		}
	}

}

void FeasibleWrenchPolytope_API::draw_friction_cone_edges(iit::dog::LegDataMap< polytope > poly,
		iit::dog::LegDataMap<Eigen::Vector3d > foot_pos,
		dwl::Color color,
		double scaling_factor,
		std::shared_ptr<dwl_rviz_plugin::DisplayInterface> display_){

	dwl::ArrowProperties noArrow(0.02, 0.0, 0.0);
	noArrow.head_length = 0.01;
	for (int leg = iit::dog::LF; leg <= iit::dog::RH; leg++) {


		display_->drawArrow(poly[leg].v_rep.block(0,0,3,1)/scaling_factor + foot_pos[leg],
				poly[leg].v_rep.block(0,1,3,1)/scaling_factor + foot_pos[leg],
				noArrow,
				color,
				"base_link");

		display_->drawArrow(poly[leg].v_rep.block(0,1,3,1)/scaling_factor + foot_pos[leg],
				poly[leg].v_rep.block(0,2,3,1)/scaling_factor + foot_pos[leg],
				noArrow,
				color,
				"base_link");

		display_->drawArrow(poly[leg].v_rep.block(0,2,3,1)/scaling_factor + foot_pos[leg],
				poly[leg].v_rep.block(0,3,3,1)/scaling_factor + foot_pos[leg],
				noArrow,
				color,
				"base_link");

		display_->drawArrow(poly[leg].v_rep.block(0,3,3,1)/scaling_factor + foot_pos[leg],
				poly[leg].v_rep.block(0,0,3,1)/scaling_factor + foot_pos[leg],
				noArrow,
				color,
				"base_link");

		display_->drawArrow(poly[leg].v_rep.block(0,4,3,1)/scaling_factor + foot_pos[leg],
				poly[leg].v_rep.block(0,0,3,1)/scaling_factor + foot_pos[leg],
				noArrow,
				color,
				"base_link");

		display_->drawArrow(poly[leg].v_rep.block(0,4,3,1)/scaling_factor + foot_pos[leg],
				poly[leg].v_rep.block(0,1,3,1)/scaling_factor + foot_pos[leg],
				noArrow,
				color,
				"base_link");

		display_->drawArrow(poly[leg].v_rep.block(0,4,3,1)/scaling_factor + foot_pos[leg],
				poly[leg].v_rep.block(0,2,3,1)/scaling_factor + foot_pos[leg],
				noArrow,
				color,
				"base_link");

		display_->drawArrow(poly[leg].v_rep.block(0,4,3,1)/scaling_factor + foot_pos[leg],
				poly[leg].v_rep.block(0,3,3,1)/scaling_factor + foot_pos[leg],
				noArrow,
				color,
				"base_link");

	}

}

void FeasibleWrenchPolytope_API::check_friction_cone_topology(boost::shared_ptr<Polytope_Rn> primal_polytope,
		topology_map & topology_map,
		topology_vec & top_vec){

	unsigned int hs_size = primal_polytope->numberOfHalfSpaces();
	unsigned int dim = primal_polytope->dimension();

	constIteratorOfListOfGeometricObjects< boost::shared_ptr<Generator_Rn> > v_iter(primal_polytope->getListOfGenerators());
	for (v_iter.begin(); v_iter.end()!=true; v_iter.next()) {
		const boost::shared_ptr<Generator_Rn> & currentGen = v_iter.current();
		double sum = 0.0;
		for(unsigned int k=0; k<dim; k++){
			sum += fabs(currentGen->getCoordinate(k));
		}
		if(sum<0.1){
			const boost::shared_ptr<Generator_Rn> & origin = currentGen;
			unsigned int origin_hs_num = origin->numberOfFacets();
			std::cout<<"the origin belongs to "<<origin_hs_num<<" facets."<<std::endl;

		}
	}

	constIteratorOfListOfGeometricObjects< boost::shared_ptr<HalfSpace_Rn> > hs_iterator(primal_polytope->getListOfHalfSpaces());
	int i;
	for (hs_iterator.begin(); hs_iterator.end()!=true; hs_iterator.next()) {
		const boost::shared_ptr<HalfSpace_Rn>& currentHalfSpace = hs_iterator.current();
		currentHalfSpace->dump(std::cout);
		std::cout<<"Constant: "<<currentHalfSpace->getConstant()<<std::endl;
		i = hs_iterator.currentIteratorNumber();
		std::cout<<"this halfspace has "<<top_vec(i)<<" vertices."<<std::endl;
		for (int j=0; j<top_vec(i);j++){
			const boost::shared_ptr<Generator_Rn> & currentGen = primal_polytope->getGenerator(topology_map(i,j));
			currentGen->dump(std::cout);
		}

	}

	std::cout<<"topology map: "<<topology_map<<std::endl;

}

void FeasibleWrenchPolytope_API::reorder_v_description(const boost::shared_ptr<Polytope_Rn> & primal_polytope, Eigen::MatrixXd & A_v_description){
	Eigen::MatrixXd reordered_v_description;
	reordered_v_description.resize(A_v_description.rows(),A_v_description.cols());
	reordered_v_description.setZero();
	unsigned int dim = primal_polytope->dimension();
	unsigned int gen_num = primal_polytope->numberOfGenerators();

	if(A_v_description.cols() == gen_num){
		constIteratorOfListOfGeometricObjects< boost::shared_ptr<Generator_Rn> > v_iter(primal_polytope->getListOfGenerators());
		for (v_iter.begin(); v_iter.end()!=true; v_iter.next()) {
			const boost::shared_ptr<Generator_Rn> & currentGen = v_iter.current();
			Eigen::VectorXd err;
			err.resize(A_v_description.cols());
			err.setZero();
			unsigned int i = v_iter.currentIteratorNumber();
			for(unsigned int k=0; k<A_v_description.cols(); k++){
				for(unsigned int j=0; j<A_v_description.rows(); j++){
					err(k) += fabs(currentGen->getCoordinate(j) -  A_v_description(j,k));
				}
			}
			double min = err.minCoeff();
			for(unsigned int k=0; k<A_v_description.cols(); k++){
				if(fabs(err(k)-min)<0.01){
					reordered_v_description.block(0,i,A_v_description.rows(),1) = A_v_description.block(0,k,A_v_description.rows(),1);
				}
			}
		}
	}else{
		std::cout<<"Warning! The polytope resulting from the Double Description has an unexpected number of generators!"<<std::endl;
	}
	std::cout<<A_v_description<<std::endl;
	A_v_description = reordered_v_description;
	std::cout<<"reordered: "<<reordered_v_description<<std::endl;
}

void FeasibleWrenchPolytope_API::set_stance_change(const iit::dog::LegDataMap<bool> previous_stance_legs,
									const iit::dog::LegDataMap<bool> new_stance_legs,
									bool & stance_flag){

	if((new_stance_legs[0]!=previous_stance_legs[0])||
			(new_stance_legs[1]!=previous_stance_legs[1])||
			(new_stance_legs[2]!=previous_stance_legs[2])||
			(new_stance_legs[3]!=previous_stance_legs[3])){
		stance_flag = true;
	}
}

void FeasibleWrenchPolytope_API::reset( CWSData & cwc,
							TLSData & awp,
							FWSData & fwp_opt,
							polytope & fwp){
	for (int leg = iit::dog::LF; leg <= iit::dog::RH; leg++) {
		awp.legs_grav_torque[leg].setZero();
	}

	fwp.hs_rep.resize(1,1);
	fwp.hs_rep.setZero();
}

void FeasibleWrenchPolytope_API::remove_average_point(const FWSData fwp_options,
										const v_description & polytope,
										const Eigen::VectorXd & wrench_gi,
										v_description & avg_polytope,
										Eigen::VectorXd & wrench_gi_avg){

    Eigen::VectorXd vertex_avg;
    wrench_gi_avg.resize(polytope.rows());
    vertex_avg.resize(polytope.rows());
    vertex_avg.setZero();
    wrench_gi_avg.setZero();
    avg_polytope.resize(polytope.rows(), polytope.cols());
    avg_polytope.setZero();

    //compute the vertices centroid
    for (int i=0; i<polytope.cols();i++)
    {
    	vertex_avg+=polytope.col(i);
    }
    vertex_avg/=(double)polytope.cols();
    //now subtract the average column wise
    avg_polytope = polytope.colwise() - vertex_avg;

    if(fwp_options.wrench_type == full_6D){
    	wrench_gi_avg = wrench_gi - vertex_avg;
    	std::cout<<"6D wrench: "<<wrench_gi_avg<<std::endl;
    }else{
    	std::cout<<"Vertex average: "<<vertex_avg<<std::endl;
    	if(fwp_options.wrench_type == linear_3D){
    		//            	std::cout<<"using linear forces only"<<std::endl;
    		wrench_gi_avg = rbd::linearPart(wrench_gi) - vertex_avg;
    	}else if(fwp_options.wrench_type == angular_3D){
    		//            	std::cout<<"using angular wrench only"<<std::endl;
    		wrench_gi_avg = rbd::angularPart(wrench_gi) - vertex_avg;
    	}
    	//        	std::cout<<"average 3D wrench: "<<wrench_gi_avg<<std::endl;
    }

}


void FeasibleWrenchPolytope_API::force_polygon_analytic_hs_rep(const v_description v_rep, hs_description & hs_descr){
	//	combos << 1.0,   1.0,  1.0,
	//			 -1.0,   1.0,  1.0,
	//			  1.0,  -1.0,  1.0,
	//			 -1.0,  -1.0,  1.0,
	//			  1.0,   1.0, -1.0,
	//			 -1.0,   1.0, -1.0,
	//			  1.0,  -1.0, -1.0,
	//			 -1.0,  -1.0, -1.0;
	//	combos.transpose();

	double b;
	hs_descr.resize(5,4); hs_descr.setZero();
	Eigen::Vector3d p0, p1, p2, vec1, vec2, normal;
	//	first plane
	p0 = v_rep.block(0,0,3,1);
	p1 = v_rep.block(0,1,3,1);
	p2 = v_rep.block(0,2,3,1);
	vec1 = p1 - p0;
	vec2 = p2 - p0;
	normal = vec1.cross(vec2);
	double l2_norm = normal.norm();
	normal /= l2_norm;
	b = normal.dot(p0);
	hs_descr.block(0,0,1,3) = normal.transpose();
	hs_descr(0,3) = b;

	//	second plane
	p0 = v_rep.block(0,0,3,1);
	p1 = v_rep.block(0,2,3,1);
	p2 = v_rep.block(0,4,3,1);
	vec1 = p1 - p0;
	vec2 = p2 - p0;
	normal = vec1.cross(vec2);
	l2_norm = normal.norm();
	normal /= l2_norm;
	b = normal.dot(p0);
	hs_descr.block(1,0,1,3) = normal.transpose();
	hs_descr(1,3) = b;

	//	third plane
	p0 = v_rep.block(0,1,3,1);
	p1 = v_rep.block(0,3,3,1);
	p2 = v_rep.block(0,5,3,1);
	vec1 = p1 - p0;
	vec2 = p2 - p0;
	normal = vec1.cross(vec2);
	l2_norm = normal.norm();
	normal /= l2_norm;
	b = normal.dot(p0);
	hs_descr.block(2,0,1,3) = normal.transpose();
	hs_descr(2,3) = b;

	//	fourth plane
	p0 = v_rep.block(0,0,3,1);
	p1 = v_rep.block(0,1,3,1);
	p2 = v_rep.block(0,4,3,1);
	vec1 = p1 - p0;
	vec2 = p2 - p0;
	normal = vec1.cross(vec2);
	l2_norm = normal.norm();
	normal /= l2_norm;
	b = normal.dot(p0);
	hs_descr.block(3,0,1,3) = normal.transpose();
	hs_descr(3,3) = b;

	//	fifth plane
	p0 = v_rep.block(0,2,3,1);
	p1 = v_rep.block(0,3,3,1);
	p2 = v_rep.block(0,6,3,1);
	vec1 = p1 - p0;
	vec2 = p2 - p0;
	normal = vec1.cross(vec2);
	l2_norm = normal.norm();
	normal /= l2_norm;
	b = normal.dot(p0);
	hs_descr.block(4,0,1,3) = normal.transpose();
	hs_descr(4,3) = b;

	for(unsigned int j = 0; j<6; j++){
		if(hs_descr(j,3)>0){
			hs_descr(j,3) = -hs_descr(j,3);
			hs_descr.block(j,0,1,3) = -hs_descr.block(j,0,1,3);
		}
	}

//	std::cout<<"analytic hs: "<<hs_descr<<std::endl;
}





