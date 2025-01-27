/*
 * CCWS_margin.h
 *
 *  Created on: April 12, 2017
 *      Author: Romeo Orsolino
 */

#ifndef FeasibleWrenchPolytope_API_H_
#define FeasibleWrenchPolytope_API_H_

#include <Eigen/Dense>
#include <math.h>
#include <ros/package.h>
#include <iit/rbd/rbd.h>
#include <iit/rbd/utils.h>
#include <iit/commons/dog/leg_data_map.h>
#include <iit/commons/dog/joint_id_tricks.h>
#include <rbdl/rbdl.h>
#include <stdio.h>
#include <string.h>

//DWL (Dynamic Whole-Body Locomotion) includes
#include <dwl/WholeBodyState.h>
#include <dwl/model/WholeBodyKinematics.h>
#include <dwl/model/WholeBodyDynamics.h>
#include <dwl/utils/RigidBodyDynamics.h>
#include <dwl/utils/Orientation.h>
#include <dwl/solver/OptimizationSolver.h>
#include <dwl/solver/IpoptNLP.h>
#include <dwl_rviz_plugin/DisplayInterface.h>

//CDD
//#include "setoper.h"
//#include "cdd_f.h"

//FWP class
#include <contact_wrench_set/mink_glpk.h>		// Minkowski sum and GLPK-based linear programming
#include <contact_wrench_set/get_weights_lp.h>
#include <contact_wrench_set/chebyshev_center.h>
#include <contact_wrench_set/testOptim.h>

// Politopix
#include "../thirdparty/politopix/trunk/politopixAPI.h"
#include "IpIpoptApplication.hpp"


#include <cassert>
#include <iostream>
#include <cstdlib>
#include <random>
#include <thread>
#include <mutex>              // std::mutex, std::unique_lock
#include <condition_variable> // std::condition_variable
#include <realtime_tools/realtime_buffer.h>

//for config files // to read in config files - include before SL includes to avoid name collisions
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>

using namespace iit;

class FeasibleWrenchPolytope_API {
public:
	typedef Eigen::Matrix<double, 4,1> Vector4D;
public:
	FeasibleWrenchPolytope_API();
	virtual ~FeasibleWrenchPolytope_API();

	void init();


	// brief Stores data required for building the Contact Wrench Set (CWS).
	struct CWSData {
		iit::dog::LegDataMap<rbd::Vector3d> stance_feet_WF;		///< Positions of stance feet in world frame
		iit::dog::LegDataMap<bool> stance_legs_flag;			///< Flag indicating which legs are in stance
		iit::dog::LegDataMap<rbd::Vector3d> normal;				///< Terrain normal at each foot
		iit::dog::LegDataMap<double> friction;					///< Friction coefficient for each contact
		iit::dog::LegDataMap<double> max_normal_force;			///< Maximum normal GRF for each foot
		double robot_mass = 92.0;								///< Robot mass
	};

	// brief Stores data for building the Torque-Limited Set (torque constraints).
	struct TLSData {
		iit::dog::LegDataMap<rbd::Vector3d> stance_feet_WF;		///< Positions of stance feet in world frame
		iit::dog::LegDataMap<rbd::Vector3d> max_torque;			///< Maximum torque for each joint or leg
		iit::dog::LegDataMap<rbd::Vector3d> legs_grav_torque;	///< Torque offset due to gravity
		Eigen::Vector3d comAWPApproxWF;							///< Approximate CoM in world frame
		Eigen::MatrixXd fixed_base_jac;							///< Jacobian for a fixed-base assumption
	};

	enum constraint{no_constraint = 0, only_friction = 1, only_actuation = 2, friction_and_actuation = 3};

	enum wrenchType{full_6D = 0 , angular_3D, linear_3D};

	// brief Configuration for building a Feasible Wrench Set (FWS), including friction, torque, and the type of wrench to compute.
	struct FWSData	{
		iit::dog::LegDataMap<rbd::Vector3d> stance_feet_WF;		///< Positions of stance feet in world frame
		iit::dog::LegDataMap<bool> stance_legs_flag;			///< Flag for stance legs
		iit::dog::LegDataMap<rbd::Vector3d> normal;				///< Contact surface normal
		iit::dog::LegDataMap<double> friction;					///< Friction coefficient per foot
		iit::dog::LegDataMap<double> max_normal_force;			///< Maximum allowed normal force
		iit::dog::LegDataMap<rbd::Vector3d> max_torque;			///< Maximum torque
		iit::dog::LegDataMap<rbd::Vector3d> legs_grav_torque;	///< Gravity-induced torque offset
		// FWP optimization, default = 0 (no optimization), CWC constraints only = 1,  AWP constraints only = 2, full FWP constraints = 3
//		int constraint_type = 3;
		
		// Specifies the type of constraints we want to apply (friction, actuation, or both)
		constraint constraint_type;

		 // Type of wrench to be computed (3D force, 3D torque, or 6D)
		wrenchType wrench_type;

		// Determines whether or not we do time-based optimization
		bool optimize_time;
	};

	typedef Eigen::MatrixXd hs_description;				///< H-representation of a polytope (A matrix of half-spaces)
	typedef Eigen::MatrixXd v_description;				///< V-representation of a polytope (list of vertices)
	typedef Eigen::MatrixXi topology_map;				///< Topology structure (edges or faces connectivity)
	typedef Eigen::VectorXi topology_vec;				///< Vector indexing for topology

	// brief Container for polytope data in both V-rep and H-rep, plus topology.
	struct polytope {
		v_description v_rep;		///< V-representation
		hs_description hs_rep;		///< H-representation
		topology_map top_map;		///< Connectivity map
		topology_vec top_vec;		///< Connectivity indexing
	};

	//--------------------------------------------------------------------------
    //                          Public API Functions
    //--------------------------------------------------------------------------



	/**
     * Computes the Zero Moment Point (ZMP) from acceleration data.
     * param r     Robot base position
     * param r_dd  Robot base linear acceleration
     * param[out] ZMP  The resulting ZMP position
     */
	void ZMP(Eigen::Vector3d r,
	                        Eigen::Vector3d r_dd,
						   Eigen::Vector3d & ZMP);

	/**
     * @brief Computes the ZMP and the distance from the stance polygon.
     * @param stance_feet_WF   Positions of stance feet in world frame
     * @param stance_legs_flag Boolean flags for stance legs
     * @param r                Robot base position
     * @param r_dd             Robot base linear acceleration
     * @param[out] ZMP         Computed Zero Moment Point
     * @param[out] ZMPdistance Euclidean distance from the support polygon
     */
	void ZMPstability(iit::dog::LegDataMap<rbd::Vector3d> stance_feet_WF,
							    iit::dog::LegDataMap<bool> stance_legs_flag,
								Eigen::Vector3d r,
	                            Eigen::Vector3d r_dd,
								Eigen::Vector3d & ZMP,
								double & ZMPdistance);

	/**
     * @brief Computes the distance from a point X to a line segment P1->P2.
     */
	void distPointToLine(Eigen::Vector3d P1,
			Eigen::Vector3d P2,
			Eigen::Vector3d X,
			double & dist);

	/**
     * @brief Computes the inertial wrench in the body frame (BF).
     * @param r_d        Linear velocity
     * @param r_dd       Linear acceleration
     * @param orient_d   Angular velocity
     * @param orient_dd  Angular acceleration
     * @param InertiaMatrix  Robot’s inertia matrix
     * @param[out] hdot_BF   The resulting 6D inertial wrench in body frame
     */
	void IntertialWrench_BF(Eigen::Vector3d r_d,
		                        Eigen::Vector3d r_dd,
		                        Eigen::Vector3d orient_d,
		                        Eigen::Vector3d orient_dd,
							   iit::rbd::Matrix66d InertiaMatrix,
		                        iit::rbd::ForceVector & hdot_BF);

	/**
     * @brief Computes the inertial wrench in the world frame (WF).
     */
	void IntertialWrench_WF(Eigen::Vector3d r,
		                        Eigen::Vector3d r_d,
		                        Eigen::Vector3d r_dd,
		                        Eigen::Vector3d orient,
		                        Eigen::Vector3d orient_d,
		                        Eigen::Vector3d orient_dd,
							   iit::rbd::Matrix66d InertiaMatrix,
							   iit::rbd::ForceVector & hdot_WF);

	/**
     * @brief Computes the gravitational wrench in the world frame, but returns nothing.
     *        (The function might store it internally.)
     */
	void GravitationalWrench_WF(Eigen::Vector3d r, CWSData cws_struct);

	/**
     * @brief Computes the gravitational wrench in the world frame, and outputs it in @p grav_wrench_W.
     */
	void GravitationalWrench_WF(Eigen::Vector3d r, CWSData cws_struct, Eigen::VectorXd & grav_wrench_W);

//	void compute_cddlib(iit::dog::LegDataMap<rbd::Vector3d> stance_feet_WF,
//        			iit::dog::LegDataMap<bool> stance_legs_flag,
//					iit::dog::LegDataMap<double> friction_coeffs,
//					iit::dog::LegDataMap<rbd::Vector3d> terrain_normals,
//					Eigen::MatrixXd & A_hs_description);
//
//	void compute_cddlib(CWSData cws_struct, Eigen::MatrixXd & A_hs_description);

	/**
     * @brief Converts from a politopix polytope to an Eigen H-representation.
     */
	void eigen_hs_description(const boost::shared_ptr<Polytope_Rn> primal_polytope, Eigen::MatrixXd & A_hs_description);

	/**
     * @brief Converts from a politopix polytope to an Eigen V-representation.
     */
	void eigen_v_description(const boost::shared_ptr<Polytope_Rn> primal_polytope, Eigen::MatrixXd & A_v_description);

	/**
     * @brief Generates the contact wrench set (CWS) for a given configuration.
     * @param cws_struct The input data structure holding stance feet, friction, etc.
     * @param[out] A_v_description The output V-representation of the CWS
     */
	void contact_wrench_set(const CWSData cws_struct, Eigen::MatrixXd & A_v_description);

	/**
     * @brief Converts an Eigen V-representation into a politopix polytope in order to
     *        obtain its H-representation.
     */
	void get_hs_description_politopix(Eigen::MatrixXd & A_v_description,
										boost::shared_ptr<Polytope_Rn> & primal_polytope);

	/**
     * @brief Overload to directly write the H-representation in an Eigen matrix.
     */
	void get_hs_description_politopix(Eigen::MatrixXd & A_v_description,
										Eigen::MatrixXd & A_hs_description);
	/**
     * @brief Analytical function to get half-space representation (HS-Rep) from a 2D polygon (V-Rep).
     */
	void force_polygon_analytic_hs_rep(const v_description v_rep, hs_description & hs_rep);

	/**
     * @brief Normalizes half-space normals to have unit magnitude.
     */
	void normalize_half_spaces(Eigen::MatrixXd & A_hs);

	/**
     * @brief Computes the intersection between a friction cone and a force polytope.
     */
	bool compute_intersection(FeasibleWrenchPolytope_API::v_description friction_cone,
											FeasibleWrenchPolytope_API::v_description force_polytope,
											const int leg_id,
											FeasibleWrenchPolytope_API::v_description & intersection);
	/**
     * @brief Builds friction cones for each contact (via discretization of the friction pyramid).
     */
	void CreateEdges(const iit::dog::LegDataMap<rbd::Vector3d> stance_feet_WF,
									const iit::dog::LegDataMap<double> friction_coeff,
									const iit::dog::LegDataMap<rbd::Vector3d> terrain_normal,
									const iit::dog::LegDataMap<double> max_normal_grf,
									iit::dog::LegDataMap< Eigen::MatrixXd > & legs_cone);

	/**
     * @brief Computes a scalar feasibility measure (margin) for a given wrench type.
     */
	void FeasibilityMargin(const FeasibleWrenchPolytope_API::wrenchType wrench_type,
							const Eigen::MatrixXd M,
							const Eigen::VectorXd wrench_gi,
							double & cws_feasibility);

	/**
     * @brief Computes the feasibility margin using a half-space-based approach.
     */
	void hs_based_margin(const FeasibleWrenchPolytope_API::wrenchType Wrench_type,
										const Eigen::MatrixXd M1,
										const Eigen::VectorXd wrench_gi,
										double & fwp_hs_margin);

	/**
     * @brief Computes rotation matrices for the edges that approximate a friction cone.
     */
	void NormalToEdgesRotation(const double half_cone_angle,
			const unsigned int edges_per_cone,
			iit::dog::LegDataMap<Eigen::Matrix3d> & R);

	/**
     * @brief Returns the norm (length) of a given 3D vector.
     */
	double GetVectorLenght(Eigen::Vector3d vec);

	/**
     * @brief Removes repeated vertices from a politopix polytope.
     */
	void remove_repeated_vertices(boost::shared_ptr<Polytope_Rn> primal_polytope);

	/**
     * @brief Removes rows from a half-space matrix if their normal is ~0.
     */
	void remove_hs_with_null_norm(const Eigen::MatrixXd A_hs, Eigen::MatrixXd & A_hs_reduced);

	/**
     * @brief Fills in the angular component of a 6D wrench polytope using a 3D polytope of forces
     *        and the foot position (for moment arm).
     */
	void fill_angular_component(const Eigen::Vector3d stance_feet_pos,
			const Eigen::MatrixXd polytope_3d,
			Eigen::MatrixXd & polytope_6d);

	/**
     * @brief Computes the 3D torque limits for each stance leg and populates a polytope struct.
     */
	void legs_torque_limits_3d(	const FeasibleWrenchPolytope_API::TLSData tls,
								iit::dog::LegDataMap< polytope > & max_lin_grf);

	/**
     * @brief Builds 3D friction cones for each stance foot, given friction coefficients and normal forces.
     */
	void legs_friction_cone_3d(const iit::dog::LegDataMap<double> friction_coeff,
			const iit::dog::LegDataMap<rbd::Vector3d> terrain_normal,
			const iit::dog::LegDataMap<double> max_normal_grf,
			iit::dog::LegDataMap< Eigen::MatrixXd > & legs_set);

	/**
     * @brief Computes the final feasible wrench set (FWS) using friction and torque limits.
     */
	bool feasibility_wrench_set(const TLSData tlws_struct,
								const CWSData cws_struct,
								const FWSData fwp_options,
								Eigen::MatrixXd & FWS_v_description);

	/**
     * @brief Computes the bounded friction polytopes (intersection of friction cones and torque limits).
     */
	bool bounded_friction_polytopes(const TLSData tlws_struct,
			const CWSData cws_struct,
			const FWSData fwp_options,
			iit::dog::LegDataMap<Eigen::MatrixXd > & bounded_fp);

	/**
     * @brief Computes the feasibility margin by solving an LP on the vertex-based representation.
     */
	void vertex_based_margin_LP(const wrenchType & wrench_type,
														const v_description & A_v, 
														const Eigen::VectorXd & wrench_gi, 
														double & fwp_margin);

	/**
     * @brief Computes the residual radius of a point wrt a set of half-spaces.
     */
	void residual_radius_LP(const hs_description A_hs, const Eigen::VectorXd wrench_gi, double & residual_radius);

	/**
     * @brief Computes the fixed-base Jacobian for each stance foot, used in torque constraints.
     */
	void get_jacobian(const int constraint_type,
						const dog::LegDataMap<Eigen::Vector3d> footPos_BF,
						Eigen::MatrixXd & fixed_base_jacobian);

//	void eigen2politopix(const iit::dog::LegDataMap<rbd::ForceVector> edges,
//			                 boost::shared_ptr<Polytope_Rn> & polytope);

//	ddf_MatrixPtr eigen2cdd(iit::dog::LegDataMap<rbd::ForceVector> edge);
//
//	void cdd2eigen(ddf_MatrixPtr A, Eigen::MatrixXd & A_eig);

//	void cdd2politopix(ddf_MatrixPtr A, boost::shared_ptr<Polytope_Rn> & poly);

//	void debugPolytope(boost::shared_ptr<Polytope_Rn> polytope);

	iit::rbd::ForceVector getGravitoInertialWrench();

	/**
     * @brief Computes the V-representation for the torque-limits-based wrench set.
     */
	void torque_limits_wrench_set(const TLSData tlws_struct,
			const Eigen::MatrixXd jacobian,
			const Eigen::Vector3d com_orientation,
			Eigen::MatrixXd & TLWS_v_description);

	/**
     * @brief Builds a 3D zonotope for joint-space boundaries, given joint limits.
     */
	void joint_space_zonotope(const Eigen::Vector3d joints_lim, Eigen::Matrix<double, 3, 8> & joint_space_zono);

	/**
     * @brief Outputs the bounded friction polytopes for the stance legs.
     */
	void get_bounded_friction_polytopes(iit::dog::LegDataMap< polytope > & bfp);

	/**
     * @brief Outputs the linear friction cones for the stance legs.
     */
	void get_linear_friction_cones(iit::dog::LegDataMap<polytope > & lfc);

	/**
     * @brief Outputs the force polytopes (intersection of friction cones and normal force limits).
     */
	void get_force_polytopes(iit::dog::LegDataMap<polytope> & fp);

	/**
     * @brief Retrieves the final feasible wrench polytope.
     */
	void get_fwp(polytope & fw_poly);

	/**
     * @brief Sorts polygon vertices in a clockwise manner for visualization or indexing.
     */
	void clock_wise_sort(const FeasibleWrenchPolytope_API::v_description v_rep,
						const FeasibleWrenchPolytope_API::topology_vec top_vec,
						FeasibleWrenchPolytope_API::topology_map & top_map);

	/**
     * @brief Builds a topological map for edges and faces from a politopix polytope.
     */
	void build_topology(const boost::shared_ptr<Polytope_Rn> primal_polytope,
						topology_map & topology,
						topology_vec & top_vec);

	/**
     * @brief Visualization function: draws polygon vertices for each leg’s polytope in RViz.
     */
	void draw_polygon_vertices(iit::dog::LegDataMap<polytope> poly,
						iit::dog::LegDataMap<Eigen::Vector3d > foot_pos,
						dwl::Color color,
						double scaling_factor,
						std::shared_ptr<dwl_rviz_plugin::DisplayInterface> display_);

	/**
     * @brief Overload to draw a polygon’s vertices with a single application point in RViz.
     */
	void draw_polygon_vertices(polytope poly,
						Eigen::Vector3d application_point,
						dwl::Color color,
						double scaling_factor,
						std::shared_ptr<dwl_rviz_plugin::DisplayInterface> display_);

	/**
     * @brief Draws edges of a polygon for each leg’s polytope in RViz.
     */
	void draw_polygon_edges(iit::dog::LegDataMap< polytope > poly,
			iit::dog::LegDataMap<Eigen::Vector3d > foot_pos,
			dwl::Color color,
			double scaling_factor,
			std::shared_ptr<dwl_rviz_plugin::DisplayInterface> display_);

	/**
     * @brief Draws friction cone edges in RViz.
     */
	void draw_friction_cone_edges(iit::dog::LegDataMap< polytope > poly,
			iit::dog::LegDataMap<Eigen::Vector3d > foot_pos,
			dwl::Color color,
			double scaling_factor,
			std::shared_ptr<dwl_rviz_plugin::DisplayInterface> display_);

	/**
     * @brief Checks the consistency of a friction cone’s topology for debugging.
     */
	void check_friction_cone_topology(boost::shared_ptr<Polytope_Rn> primal_polytope,
			topology_map & topology,
			topology_vec & top_vec);

	/**
     * @brief Reorders vertices from a politopix polytope for an Eigen-based V-representation.
     */
	void reorder_v_description(const boost::shared_ptr<Polytope_Rn> & primal_polytope,
								Eigen::MatrixXd & A_v_description);

	/**
     * @brief Updates a flag if stance legs have changed, possibly triggering a re-computation.
     */
	void set_stance_change(const iit::dog::LegDataMap<bool> previous_stance_legs,
										const iit::dog::LegDataMap<bool> new_stance_legs,
										bool & stance_flag);

	/**
     * @brief Resets the internal polytope structures and data fields.
     */
	void reset(CWSData & cwc,
			TLSData & awp,
			FWSData & fwp_opt,
			polytope & fwp);

	/**
     * @brief Removes the average point (centroid) from the polytope representation, if desired.
     */
	void remove_average_point(const FWSData fwp_options,
			const v_description & polytope,
			const Eigen::VectorXd & wrench_gi,
			v_description & avg_polytope,
			Eigen::VectorXd & wrench_gi_avg);

	void kill();

	void setPrintAll(bool print_all_flag){
	    this->print_all = print_all_flag;
	}

	Eigen::MatrixXd A_matrix;

	typedef realtime_tools::RealtimeBuffer<Eigen::VectorXd> cws_buffer;
	typedef realtime_tools::RealtimeBuffer<FeasibleWrenchPolytope_API::CWSData> cws_buffer_read;
	typedef std::mutex cws_mtx;
	typedef std::condition_variable cws_condv;

//	void getPointsNum(unsigned int & pointsNumber);

private:

	//--------------------------------------------------------------------------
    //                           Private Members
    //--------------------------------------------------------------------------

	double g0;			///< Gravity constant
	double mass;		///< Robot mass

    iit::dog::JointState q;		///< Joint angles (or positions) buffer
	double TOL = 10^-3;			///< Numerical tolerance

//	std::string param_file = ros::package::getPath(std::string("contact_wrench_set")) + "/config/cws_options.ini";
	//config files
//	boost::property_tree::ptree config_;
//	bool print_friction_edges = config_.get<double>("Options.print_friction_cone_edges");
	bool print_friction_edges = false;
	// in case that only a 3D dynamics is used (therefore for use_6d_dynamics = false) you can
	// chose between using the only the linear wrench or only the angular wrench
	bool use_linear_wrench = true;
	bool print_all = false;
	bool use_torque_set;
	bool use_simplified_torque_limits_set = false;

//	Generator_Rn vx = Generator_Rn(cardinality);
	iit::dog::LegDataMap<Eigen::Vector3d> footPos_WF;												///< Foot positions in WF
	RigidBodyDynamics::Math::SpatialVector vel = RigidBodyDynamics::Math::SpatialVector::Zero();	///< Full robot velocity
	RigidBodyDynamics::Math::SpatialVector acc = RigidBodyDynamics::Math::SpatialVector::Zero();	///< Full robot acceleration
	iit::dog::LegDataMap<double> max_normal_force;													///< Maximum normal forces
	iit::dog::LegDataMap<double> normal_force_projection;
	iit::dog::LegDataMap<double> half_cone_angle;
	iit::dog::LegDataMap<Eigen::Matrix3d> R;														///< Rotation matrices for each friction cone
	iit::dog::LegDataMap< polytope > bounded_friction_poly,force_polytopes;
	iit::dog::LegDataMap<Eigen::MatrixXd > linear_friction_cones;
	iit::dog::LegDataMap<iit::dog::LegDataMap<rbd::Vector3d> > friction_cone_verteces;
	iit::dog::LegDataMap<iit::dog::LegDataMap<rbd::Vector3d> > friction_cone_edges;
	iit::dog::LegDataMap<iit::dog::LegDataMap<rbd::ForceVector> > cws_pyramid;

	Eigen::MatrixXd A_hs;																			///< Half-space matrix
	topology_map lfc_top_map, bfc_top_map, fp_top_map, fwp_top_map;
	topology_vec lfc_top_vec, bfc_top_vec, fp_top_vec, fwp_top_vec;
	//std::shared_ptr< dwl::solver::OptimizationSolver> solver2;

	// Chebyshev center or other optimization-based methods
	dwl::solver::OptimizationSolver* margin_solver = new dwl::solver::IpoptNLP();
    OptWeights opt_weights;
	ChebyshevCenter cheb_center;
	std::shared_ptr< dwl::solver::OptimizationSolver> solver;

	// Polytopes for contact-wrench (cwc), actuation-wrench (awp), final feasible-wrench (fwp)
	polytope cwc, awp, fwp;

	int stance_num;
	unsigned int cardinality = 6;
	unsigned int pointsNum = 5;
	double dist, num, denum;
	double margin, old_cws_feasibility;
	rbd::ForceVector inertial_wrench_WF, grav_wrench_WF;
//	ddf_rowindex newpos;
//	ddf_rowset impl_lin,redset;

	/** @brief Kinematical model */
	dwl::model::WholeBodyKinematics wb_kin;
	/** @brief Kinematical model */
	dwl::model::WholeBodyDynamics wb_dyn;
	/** @brief Actual whole-body state information */
	dwl::WholeBodyState current_wbs;
	/** Floating-base system */
	dwl::model::FloatingBaseSystem fbs;

	// Resetting the system from the hyq urdf file
	std::string model_;
	string robot_;
//	dwl::rbd::BodyVector3d ik_pos;
//	dwl::rbd::BodySelector feet_names;
//	std::vector<unsigned int> foot_id_map_;
};

#endif /* FeasibleWrenchPolytope_API_H_ */





